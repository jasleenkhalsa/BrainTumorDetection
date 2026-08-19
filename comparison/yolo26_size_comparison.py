from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VAL_IMAGES = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "val"
    / "images"
)

VAL_LABELS = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "val"
    / "labels"
)

YOLO26_MODEL = (
    PROJECT_ROOT
    / "runs"
    / "YOLO26_Baseline"
    / "weights"
    / "best.pt"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "comparison"
    / "Figure8_YOLO26_Size_Comparison.png"
)


# ============================================================
# MODEL / MATCHING SETTINGS
# ============================================================

IMAGE_SIZE = 640

CONF_THRESHOLD = 0.25

NMS_IOU_THRESHOLD = 0.45

MATCH_IOU_THRESHOLD = 0.50


# ============================================================
# SIZE DEFINITIONS
#
# These are study-specific definitions based on the relative
# bounding-box area, NOT clinical tumor staging.
# ============================================================

SMALL_MAX = 0.01       # < 1%

MEDIUM_MAX = 0.03      # 1% to 3%

# Large = > 3%


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary",
}


# ============================================================
# DISPLAY SETTINGS
# ============================================================

PANEL_WIDTH = 520

HEADER_HEIGHT = 125


# ============================================================
# LABEL READING
# ============================================================

def read_yolo_labels(label_path: Path) -> list[dict]:
    """
    Read YOLO-format annotations.

    Format:
        class_id x_center y_center width height

    All coordinates are normalized between 0 and 1.
    """

    annotations = []

    if not label_path.exists():
        return annotations

    lines = label_path.read_text(
        encoding="utf-8"
    ).splitlines()

    for index, line in enumerate(lines):
        parts = line.strip().split()

        if len(parts) != 5:
            continue

        class_id = int(float(parts[0]))

        x_center = float(parts[1])
        y_center = float(parts[2])

        width = float(parts[3])
        height = float(parts[4])

        relative_area = width * height

        annotations.append(
            {
                "index": index,
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
                "relative_area": relative_area,
            }
        )

    return annotations


# ============================================================
# SIZE CLASSIFICATION
# ============================================================

def get_size_category(relative_area: float) -> str:
    """
    Categorize a lesion according to relative bounding-box area.
    """

    if relative_area < SMALL_MAX:
        return "Small"

    if relative_area <= MEDIUM_MAX:
        return "Medium"

    return "Large"


# ============================================================
# CONVERT YOLO BOX TO XYXY
# ============================================================

def yolo_to_xyxy(
    annotation: dict,
    image_width: int,
    image_height: int,
) -> list[float]:

    x_center = annotation["x_center"] * image_width
    y_center = annotation["y_center"] * image_height

    width = annotation["width"] * image_width
    height = annotation["height"] * image_height

    x1 = x_center - width / 2
    y1 = y_center - height / 2

    x2 = x_center + width / 2
    y2 = y_center + height / 2

    return [
        x1,
        y1,
        x2,
        y2,
    ]


# ============================================================
# IOU
# ============================================================

def calculate_iou(
    box_a: list[float],
    box_b: list[float],
) -> float:

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)

    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(
        0.0,
        inter_x2 - inter_x1,
    )

    inter_height = max(
        0.0,
        inter_y2 - inter_y1,
    )

    intersection = (
        inter_width
        * inter_height
    )

    area_a = (
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# MATCH YOLO26 PREDICTION TO GT
# ============================================================

def find_best_prediction(
    result,
    ground_truth: dict,
    image_width: int,
    image_height: int,
) -> dict | None:

    if (
        result.boxes is None
        or len(result.boxes) == 0
    ):
        return None

    gt_box = yolo_to_xyxy(
        ground_truth,
        image_width,
        image_height,
    )

    best_match = None

    best_iou = 0.0

    for box in result.boxes:

        predicted_class = int(
            box.cls.item()
        )

        # Only compare class-correct predictions.
        if (
            predicted_class
            != ground_truth["class_id"]
        ):
            continue

        predicted_box = [
            float(value)
            for value
            in box.xyxy[0].tolist()
        ]

        iou = calculate_iou(
            gt_box,
            predicted_box,
        )

        if iou > best_iou:

            best_iou = iou

            best_match = {
                "bbox": predicted_box,
                "confidence": float(
                    box.conf.item()
                ),
                "iou": iou,
            }

    return best_match


# ============================================================
# SCORE REPRESENTATIVE EXAMPLE
# ============================================================

def candidate_score(candidate: dict) -> float:
    """
    Prefer examples with:
    - high IoU
    - high confidence

    IoU is weighted slightly more heavily.
    """

    return (
        candidate["iou"] * 0.7
        + candidate["confidence"] * 0.3
    )


# ============================================================
# SEARCH VALIDATION SET
# ============================================================

def find_representative_examples(
    model: YOLO,
) -> dict[str, dict]:

    best_examples = {
        "Small": None,
        "Medium": None,
        "Large": None,
    }

    image_paths = sorted(
        [
            *VAL_IMAGES.glob("*.jpg"),
            *VAL_IMAGES.glob("*.jpeg"),
            *VAL_IMAGES.glob("*.png"),
        ]
    )

    print(
        f"Validation images found: "
        f"{len(image_paths)}"
    )

    for image_number, image_path in enumerate(
        image_paths,
        start=1,
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        image_height, image_width = (
            image.shape[:2]
        )

        label_path = (
            VAL_LABELS
            / f"{image_path.stem}.txt"
        )

        annotations = read_yolo_labels(
            label_path
        )

        # Skip images with no annotations.
        if not annotations:
            continue

        # Skip the No Tumor class.
        tumor_annotations = [
            annotation
            for annotation in annotations
            if annotation["class_id"] != 2
        ]

        if not tumor_annotations:
            continue

        result = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=CONF_THRESHOLD,
            iou=NMS_IOU_THRESHOLD,
            device=0,
            verbose=False,
        )[0]

        for annotation in tumor_annotations:

            size_category = (
                get_size_category(
                    annotation[
                        "relative_area"
                    ]
                )
            )

            match = find_best_prediction(
                result=result,
                ground_truth=annotation,
                image_width=image_width,
                image_height=image_height,
            )

            if match is None:
                continue

            # Detection must satisfy IoU >= 0.50.
            if (
                match["iou"]
                < MATCH_IOU_THRESHOLD
            ):
                continue

            candidate = {
                "image_path": image_path,
                "annotation": annotation,
                "bbox_gt": yolo_to_xyxy(
                    annotation,
                    image_width,
                    image_height,
                ),
                "bbox_pred": match["bbox"],
                "confidence": (
                    match["confidence"]
                ),
                "iou": match["iou"],
                "class_name": CLASS_NAMES.get(
                    annotation["class_id"],
                    str(
                        annotation[
                            "class_id"
                        ]
                    ),
                ),
                "relative_area": (
                    annotation[
                        "relative_area"
                    ]
                ),
            }

            current_best = (
                best_examples[
                    size_category
                ]
            )

            if current_best is None:

                best_examples[
                    size_category
                ] = candidate

            elif (
                candidate_score(candidate)
                >
                candidate_score(
                    current_best
                )
            ):

                best_examples[
                    size_category
                ] = candidate

        if image_number % 100 == 0:

            print(
                f"Processed "
                f"{image_number}/"
                f"{len(image_paths)}"
            )

    return best_examples


# ============================================================
# DRAW RESULT
# ============================================================

def draw_detection_panel(
    candidate: dict,
    size_label: str,
) -> np.ndarray:

    image = cv2.imread(
        str(
            candidate[
                "image_path"
            ]
        )
    )

    output = image.copy()

    # --------------------------------------------------------
    # Ground truth box
    # --------------------------------------------------------

    gx1, gy1, gx2, gy2 = [
        int(round(value))
        for value
        in candidate["bbox_gt"]
    ]

    cv2.rectangle(
        output,
        (gx1, gy1),
        (gx2, gy2),
        (0, 215, 255),
        2,
    )

    # --------------------------------------------------------
    # YOLO26 prediction
    # --------------------------------------------------------

    px1, py1, px2, py2 = [
        int(round(value))
        for value
        in candidate["bbox_pred"]
    ]

    cv2.rectangle(
        output,
        (px1, py1),
        (px2, py2),
        (0, 180, 0),
        3,
    )

    prediction_text = (
        f"{candidate['class_name']} "
        f"{candidate['confidence']:.3f}"
    )

    cv2.putText(
        output,
        prediction_text,
        (
            max(10, px1),
            max(25, py1 - 8),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 180, 0),
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    height, width = (
        output.shape[:2]
    )

    target_height = int(
        height
        * PANEL_WIDTH
        / width
    )

    output = cv2.resize(
        output,
        (
            PANEL_WIDTH,
            target_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    # --------------------------------------------------------
    # Add header
    # --------------------------------------------------------

    panel = cv2.copyMakeBorder(
        output,
        HEADER_HEIGHT,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    area_percent = (
        candidate[
            "relative_area"
        ]
        * 100
    )

    # Main title
    cv2.putText(
        panel,
        f"{size_label} Tumor",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (45, 45, 45),
        2,
        cv2.LINE_AA,
    )

    # Class
    cv2.putText(
        panel,
        (
            f"Class: "
            f"{candidate['class_name']}"
        ),
        (18, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (55, 55, 55),
        1,
        cv2.LINE_AA,
    )

    # Area
    cv2.putText(
        panel,
        (
            f"Relative area: "
            f"{area_percent:.3f}%"
        ),
        (18, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (55, 55, 55),
        1,
        cv2.LINE_AA,
    )

    # Confidence / IoU
    cv2.putText(
        panel,
        (
            f"Confidence: "
            f"{candidate['confidence']:.3f}"
            f" | IoU: "
            f"{candidate['iou']:.3f}"
        ),
        (18, 106),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (55, 55, 55),
        1,
        cv2.LINE_AA,
    )

    return panel


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    for path in [
        VAL_IMAGES,
        VAL_LABELS,
        YOLO26_MODEL,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Required path "
                f"not found:\n{path}"
            )

    print("=" * 70)

    print(
        "YOLO26 Tumor Size Detection Analysis"
    )

    print("=" * 70)

    print(
        "Size definitions:"
    )

    print(
        "Small  : < 1% image area"
    )

    print(
        "Medium : 1% - 3% image area"
    )

    print(
        "Large  : > 3% image area"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load YOLO26
    # --------------------------------------------------------

    print(
        "\nLoading YOLO26..."
    )

    model = YOLO(
        str(YOLO26_MODEL)
    )

    # --------------------------------------------------------
    # Find best example for each size
    # --------------------------------------------------------

    examples = (
        find_representative_examples(
            model
        )
    )

    print(
        "\nSelected examples:"
    )

    for size in [
        "Small",
        "Medium",
        "Large",
    ]:

        example = examples[size]

        if example is None:

            print(
                f"{size}: "
                "No qualifying example found."
            )

            continue

        print(
            f"\n{size}"
        )

        print(
            f"Image: "
            f"{example['image_path'].name}"
        )

        print(
            f"Class: "
            f"{example['class_name']}"
        )

        print(
            f"Relative area: "
            f"{example['relative_area'] * 100:.4f}%"
        )

        print(
            f"Confidence: "
            f"{example['confidence']:.4f}"
        )

        print(
            f"IoU: "
            f"{example['iou']:.4f}"
        )

    # --------------------------------------------------------
    # Ensure all categories exist
    # --------------------------------------------------------

    if any(
        examples[size] is None
        for size in [
            "Small",
            "Medium",
            "Large",
        ]
    ):

        raise RuntimeError(
            "Could not find qualifying examples "
            "for all three size categories."
        )

    # --------------------------------------------------------
    # Generate three panels
    # --------------------------------------------------------

    panels = []

    for size in [
        "Small",
        "Medium",
        "Large",
    ]:

        panel = (
            draw_detection_panel(
                examples[size],
                size,
            )
        )

        panels.append(panel)

    # --------------------------------------------------------
    # Equalize heights
    # --------------------------------------------------------

    minimum_height = min(
        panel.shape[0]
        for panel in panels
    )

    resized_panels = []

    for panel in panels:

        resized = cv2.resize(
            panel,
            (
                PANEL_WIDTH,
                minimum_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

        resized = cv2.copyMakeBorder(
            resized,
            2,
            2,
            2,
            2,
            cv2.BORDER_CONSTANT,
            value=(210, 210, 210),
        )

        resized_panels.append(
            resized
        )

    # --------------------------------------------------------
    # Spacing between panels
    # --------------------------------------------------------

    spacer = np.full(
        (
            resized_panels[0].shape[0],
            12,
            3,
        ),
        255,
        dtype=np.uint8,
    )

    final_figure = cv2.hconcat(
        [
            resized_panels[0],
            spacer,
            resized_panels[1],
            spacer,
            resized_panels[2],
        ]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(OUTPUT_PATH),
        final_figure,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "Figure generated successfully."
    )

    print(
        f"Saved to:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        "=" * 70
    )

    cv2.imshow(
        "YOLO26 Tumor Size Comparison",
        final_figure,
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()