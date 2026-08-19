from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "val"
    / "images"
    / "gg (136).jpg"
)

LABEL_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "val"
    / "labels"
    / "gg (136).txt"
)

MODEL_PATHS = {
    "YOLOv8": (
        PROJECT_ROOT
        / "runs"
        / "YOLOv8_Baseline"
        / "weights"
        / "best.pt"
    ),
    "YOLO12": (
        PROJECT_ROOT
        / "runs"
        / "YOLO12_Baseline"
        / "weights"
        / "best.pt"
    ),
    "YOLO26": (
        PROJECT_ROOT
        / "runs"
        / "YOLO26_Baseline"
        / "weights"
        / "best.pt"
    ),
}

OUTPUT_PATH = (
    PROJECT_ROOT
    / "comparison"
    / "Figure7_gg136_small_tumor_comparison.png"
)


# ============================================================
# Inference and matching settings
# ============================================================

IMAGE_SIZE = 640
CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45
MATCH_IOU_THRESHOLD = 0.50

# Fourth annotation in gg (136).txt:
# 0 0.536133 0.473633 0.037109 0.021484
TARGET_LABEL_INDEX = 4

PANEL_WIDTH = 520
HEADER_HEIGHT = 105


# ============================================================
# Utility functions
# ============================================================

def read_target_ground_truth(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> dict:
    """
    Read the selected YOLO-format annotation and convert it
    from normalized xywh to pixel xyxy coordinates.
    """

    lines = [
        line.strip()
        for line in label_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if TARGET_LABEL_INDEX > len(lines):
        raise ValueError(
            f"Target annotation {TARGET_LABEL_INDEX} does not exist."
        )

    parts = lines[TARGET_LABEL_INDEX - 1].split()

    if len(parts) != 5:
        raise ValueError(
            f"Invalid YOLO label line: {lines[TARGET_LABEL_INDEX - 1]}"
        )

    class_id = int(float(parts[0]))
    x_center = float(parts[1])
    y_center = float(parts[2])
    width = float(parts[3])
    height = float(parts[4])

    x1 = (x_center - width / 2) * image_width
    y1 = (y_center - height / 2) * image_height
    x2 = (x_center + width / 2) * image_width
    y2 = (y_center + height / 2) * image_height

    return {
        "class_id": class_id,
        "bbox": [x1, y1, x2, y2],
        "relative_area": width * height,
    }


def calculate_iou(
    box_a: list[float],
    box_b: list[float],
) -> float:
    """Calculate Intersection over Union between two xyxy boxes."""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)

    intersection = inter_width * inter_height

    area_a = max(0.0, ax2 - ax1) * max(
        0.0,
        ay2 - ay1,
    )

    area_b = max(0.0, bx2 - bx1) * max(
        0.0,
        by2 - by1,
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def find_best_class_match(
    result,
    target_ground_truth: dict,
) -> dict | None:
    """
    Find the prediction of the same class having the highest
    IoU with the selected ground-truth box.
    """

    if result.boxes is None or len(result.boxes) == 0:
        return None

    best_match = None
    best_iou = 0.0

    for box in result.boxes:
        predicted_class = int(box.cls.item())

        if predicted_class != target_ground_truth["class_id"]:
            continue

        predicted_bbox = [
            float(value)
            for value in box.xyxy[0].tolist()
        ]

        iou = calculate_iou(
            predicted_bbox,
            target_ground_truth["bbox"],
        )

        if iou > best_iou:
            best_iou = iou

            best_match = {
                "bbox": predicted_bbox,
                "confidence": float(box.conf.item()),
                "iou": iou,
            }

    return best_match


def resize_to_width(
    image: np.ndarray,
    target_width: int,
) -> np.ndarray:
    """Resize while preserving aspect ratio."""

    height, width = image.shape[:2]

    target_height = int(
        height * target_width / width
    )

    return cv2.resize(
        image,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )


def draw_ground_truth(
    image: np.ndarray,
    ground_truth: dict,
) -> np.ndarray:
    """Draw only the selected tiny ground-truth box."""

    output = image.copy()

    x1, y1, x2, y2 = [
        int(round(value))
        for value in ground_truth["bbox"]
    ]

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (0, 255, 255),
        3,
    )

    cv2.putText(
        output,
        "Target small glioma",
        (max(10, x1 - 45), max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


def draw_selected_prediction(
    image: np.ndarray,
    match: dict | None,
    model_name: str,
) -> np.ndarray:
    """
    Draw only the prediction that best matches the selected
    ground-truth lesion. This avoids clutter from other boxes.
    """

    output = image.copy()

    if match is None:
        cv2.putText(
            output,
            "No matching detection",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 220),
            2,
            cv2.LINE_AA,
        )

        return output

    x1, y1, x2, y2 = [
        int(round(value))
        for value in match["bbox"]
    ]

    detected = match["iou"] >= MATCH_IOU_THRESHOLD

    box_color = (
        (0, 170, 0)
        if detected
        else (0, 0, 220)
    )

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        box_color,
        3,
    )

    label = (
        f"{model_name}: "
        f"{match['confidence']:.3f}"
    )

    cv2.putText(
        output,
        label,
        (max(10, x1), max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        box_color,
        2,
        cv2.LINE_AA,
    )

    return output


def add_header(
    image: np.ndarray,
    title: str,
    subtitle_lines: list[str],
    title_color: tuple[int, int, int],
) -> np.ndarray:
    """Add a clean white publication-style header above a panel."""

    panel = cv2.copyMakeBorder(
        image,
        HEADER_HEIGHT,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    cv2.putText(
        panel,
        title,
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        title_color,
        2,
        cv2.LINE_AA,
    )

    y_position = 64

    for line in subtitle_lines:
        cv2.putText(
            panel,
            line,
            (18, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (45, 45, 45),
            1,
            cv2.LINE_AA,
        )

        y_position += 24

    return panel


def add_panel_border(
    image: np.ndarray,
) -> np.ndarray:
    """Add a thin light border around each panel."""

    return cv2.copyMakeBorder(
        image,
        2,
        2,
        2,
        2,
        cv2.BORDER_CONSTANT,
        value=(210, 210, 210),
    )


# ============================================================
# Main workflow
# ============================================================

def main() -> None:
    required_paths = [
        IMAGE_PATH,
        LABEL_PATH,
        *MODEL_PATHS.values(),
    ]

    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{required_path}"
            )

    original = cv2.imread(str(IMAGE_PATH))

    if original is None:
        raise ValueError(
            f"Could not read image:\n{IMAGE_PATH}"
        )

    image_height, image_width = original.shape[:2]

    ground_truth = read_target_ground_truth(
        label_path=LABEL_PATH,
        image_width=image_width,
        image_height=image_height,
    )

    print("Loading models and running predictions...")

    results = {}
    matches = {}

    for model_name, model_path in MODEL_PATHS.items():
        print(f"Running {model_name}...")

        model = YOLO(str(model_path))

        result = model.predict(
            source=str(IMAGE_PATH),
            imgsz=IMAGE_SIZE,
            conf=CONF_THRESHOLD,
            iou=NMS_IOU_THRESHOLD,
            device=0,
            verbose=False,
        )[0]

        results[model_name] = result

        matches[model_name] = find_best_class_match(
            result,
            ground_truth,
        )

    area_percent = (
        ground_truth["relative_area"] * 100
    )

    # --------------------------------------------------------
    # Ground-truth panel
    # --------------------------------------------------------

    ground_truth_image = draw_ground_truth(
        original,
        ground_truth,
    )

    ground_truth_image = resize_to_width(
        ground_truth_image,
        PANEL_WIDTH,
    )

    ground_truth_panel = add_header(
        ground_truth_image,
        title="Ground Truth",
        subtitle_lines=[
            "Class: Glioma",
            f"Relative area: {area_percent:.4f}%",
        ],
        title_color=(95, 70, 20),
    )

    # --------------------------------------------------------
    # Model panels
    # --------------------------------------------------------

    model_panels = []

    title_colors = {
        "YOLOv8": (0, 0, 210),
        "YOLO12": (190, 95, 0),
        "YOLO26": (0, 130, 0),
    }

    for model_name in [
        "YOLOv8",
        "YOLO12",
        "YOLO26",
    ]:
        match = matches[model_name]

        model_image = draw_selected_prediction(
            original,
            match,
            model_name,
        )

        model_image = resize_to_width(
            model_image,
            PANEL_WIDTH,
        )

        if match is None:
            subtitle_lines = [
                "Status: Missed",
                "IoU: 0.000",
            ]

        else:
            status = (
                "Detected"
                if match["iou"] >= MATCH_IOU_THRESHOLD
                else "Missed"
            )

            subtitle_lines = [
                f"Status: {status}",
                (
                    f"Confidence: "
                    f"{match['confidence']:.3f} | "
                    f"IoU: {match['iou']:.3f}"
                ),
            ]

        model_panel = add_header(
            model_image,
            title=model_name,
            subtitle_lines=subtitle_lines,
            title_color=title_colors[model_name],
        )

        model_panels.append(model_panel)

    # --------------------------------------------------------
    # Equalize panel heights
    # --------------------------------------------------------

    all_panels = [
        ground_truth_panel,
        *model_panels,
    ]

    minimum_height = min(
        panel.shape[0]
        for panel in all_panels
    )

    resized_panels = []

    for panel in all_panels:
        panel = cv2.resize(
            panel,
            (
                PANEL_WIDTH,
                minimum_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

        resized_panels.append(
            add_panel_border(panel)
        )

    # Add small white spacing between panels
    spacer = np.full(
        (
            resized_panels[0].shape[0],
            10,
            3,
        ),
        255,
        dtype=np.uint8,
    )

    comparison = cv2.hconcat([
        resized_panels[0],
        spacer,
        resized_panels[1],
        spacer,
        resized_panels[2],
        spacer,
        resized_panels[3],
    ])

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved = cv2.imwrite(
        str(OUTPUT_PATH),
        comparison,
    )

    if not saved:
        raise RuntimeError(
            f"Could not save output:\n{OUTPUT_PATH}"
        )

    # --------------------------------------------------------
    # Print measured values
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Figure generated successfully")
    print("=" * 70)
    print(f"Image: {IMAGE_PATH.name}")
    print(f"Ground-truth area: {area_percent:.4f}%")

    for model_name in [
        "YOLOv8",
        "YOLO12",
        "YOLO26",
    ]:
        match = matches[model_name]

        if match is None:
            print(
                f"{model_name}: "
                "No class-correct matching prediction"
            )
        else:
            status = (
                "Detected"
                if match["iou"] >= MATCH_IOU_THRESHOLD
                else "Missed"
            )

            print(
                f"{model_name}: "
                f"{status}, "
                f"confidence={match['confidence']:.4f}, "
                f"IoU={match['iou']:.4f}"
            )

    print(f"\nSaved to:\n{OUTPUT_PATH}")

    cv2.imshow(
        "YOLO Small Tumor Comparison",
        comparison,
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()