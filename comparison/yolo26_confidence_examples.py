from pathlib import Path
import csv

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

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "YOLO26_Baseline"
    / "weights"
    / "best.pt"
)

OUTPUT_IMAGE = (
    PROJECT_ROOT
    / "comparison"
    / "Figure_YOLO26_Confidence_Examples.png"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "comparison"
    / "YOLO26_Confidence_Examples.csv"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 640

# Minimum confidence used during inference.
# Keep this lower than 0.85 because we search around 85/90/95%.
PREDICTION_CONFIDENCE = 0.20

# Prediction must overlap the correct ground truth by at least this much.
MATCH_IOU_THRESHOLD = 0.50

# Desired confidence examples.
TARGET_CONFIDENCES = [
    0.85,
    0.90,
    0.95,
]


# ============================================================
# DATASET CLASSES
# ============================================================

CLASS_NAMES = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary",
}

# We only want the three actual tumor types.
TUMOR_CLASSES = {
    0: "Glioma",
    1: "Meningioma",
    3: "Pituitary",
}


# ============================================================
# DISPLAY SETTINGS
# ============================================================

PANEL_WIDTH = 430
PANEL_HEIGHT = 430
HEADER_HEIGHT = 120

# Bounding-box color: green.
BOX_COLOR = (0, 200, 0)


# ============================================================
# READ YOLO GROUND-TRUTH LABELS
# ============================================================

def read_labels(label_path):
    """
    YOLO detection label format:

    class_id x_center y_center width height

    Coordinates are normalized between 0 and 1.
    """

    annotations = []

    if not label_path.exists():
        return annotations

    with open(label_path, "r", encoding="utf-8") as file:
        for line in file:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(float(parts[0]))

            x_center = float(parts[1])
            y_center = float(parts[2])

            width = float(parts[3])
            height = float(parts[4])

            annotations.append(
                {
                    "class_id": class_id,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                }
            )

    return annotations


# ============================================================
# CONVERT NORMALIZED YOLO BOX TO PIXEL XYXY
# ============================================================

def ground_truth_xyxy(annotation, image_width, image_height):

    x_center = annotation["x_center"] * image_width
    y_center = annotation["y_center"] * image_height

    box_width = annotation["width"] * image_width
    box_height = annotation["height"] * image_height

    x1 = x_center - box_width / 2
    y1 = y_center - box_height / 2

    x2 = x_center + box_width / 2
    y2 = y_center + box_height / 2

    return [x1, y1, x2, y2]


# ============================================================
# IOU
# ============================================================

def calculate_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)

    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0,
        intersection_x2 - intersection_x1,
    )

    intersection_height = max(
        0,
        intersection_y2 - intersection_y1,
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    area_a = (
        max(0, ax2 - ax1)
        * max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1)
        * max(0, by2 - by1)
    )

    union = (
        area_a
        + area_b
        - intersection_area
    )

    if union <= 0:
        return 0.0

    return intersection_area / union


# ============================================================
# GET VALID CLASS-CORRECT PREDICTIONS
# ============================================================

def get_valid_predictions(
    result,
    annotations,
    image_width,
    image_height,
):

    valid_predictions = []

    if result.boxes is None:
        return valid_predictions

    for prediction in result.boxes:

        predicted_class = int(
            prediction.cls.item()
        )

        # Ignore No Tumor.
        if predicted_class not in TUMOR_CLASSES:
            continue

        predicted_confidence = float(
            prediction.conf.item()
        )

        predicted_box = [
            float(value)
            for value
            in prediction.xyxy[0].tolist()
        ]

        best_iou = 0.0

        # Compare prediction only to same-class GT boxes.
        for annotation in annotations:

            if (
                annotation["class_id"]
                != predicted_class
            ):
                continue

            gt_box = ground_truth_xyxy(
                annotation,
                image_width,
                image_height,
            )

            iou = calculate_iou(
                predicted_box,
                gt_box,
            )

            if iou > best_iou:
                best_iou = iou

        # Keep only correct/localized detections.
        if best_iou >= MATCH_IOU_THRESHOLD:

            valid_predictions.append(
                {
                    "class_id": predicted_class,
                    "class_name": TUMOR_CLASSES[
                        predicted_class
                    ],
                    "confidence": predicted_confidence,
                    "bbox": predicted_box,
                    "iou": best_iou,
                }
            )

    return valid_predictions


# ============================================================
# SEARCH VALIDATION SET
# ============================================================

def find_examples(model):

    # Store ALL valid candidates first.
    candidates = {
        class_id: []
        for class_id
        in TUMOR_CLASSES
    }

    image_paths = sorted(
        list(VAL_IMAGES.glob("*.jpg"))
        + list(VAL_IMAGES.glob("*.jpeg"))
        + list(VAL_IMAGES.glob("*.png"))
    )

    print(
        f"Validation images found: "
        f"{len(image_paths)}"
    )

    for index, image_path in enumerate(
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

        annotations = read_labels(
            label_path
        )

        # Skip images without tumor annotations.
        tumor_annotations = [
            annotation
            for annotation in annotations
            if annotation["class_id"]
            in TUMOR_CLASSES
        ]

        if not tumor_annotations:
            continue

        result = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=PREDICTION_CONFIDENCE,
            device=0,
            verbose=False,
        )[0]

        valid_predictions = (
            get_valid_predictions(
                result,
                tumor_annotations,
                image_width,
                image_height,
            )
        )

        for prediction in valid_predictions:

            prediction["image_path"] = (
                image_path
            )

            candidates[
                prediction["class_id"]
            ].append(
                prediction
            )

        if index % 100 == 0:

            print(
                f"Processed "
                f"{index}/"
                f"{len(image_paths)}"
            )

    # --------------------------------------------------------
    # Select nearest confidence example for each target.
    # --------------------------------------------------------

    selected = {}

    for class_id, class_name in (
        TUMOR_CLASSES.items()
    ):

        class_candidates = candidates[
            class_id
        ]

        selected[class_id] = {}

        used_images = set()

        for target in TARGET_CONFIDENCES:

            # Prefer a different MRI for every target.
            available = [
                candidate
                for candidate
                in class_candidates
                if candidate[
                    "image_path"
                ].name
                not in used_images
            ]

            # If not enough unique images,
            # allow reuse as a fallback.
            if not available:
                available = class_candidates

            if not available:

                selected[class_id][
                    target
                ] = None

                continue

            best = min(
                available,
                key=lambda item: abs(
                    item["confidence"]
                    - target
                ),
            )

            selected[class_id][
                target
            ] = best

            used_images.add(
                best["image_path"].name
            )

    return selected


# ============================================================
# CREATE PANEL
# ============================================================

def create_panel(
    example,
    target_confidence,
):

    image = cv2.imread(
        str(example["image_path"])
    )

    if image is None:
        raise RuntimeError(
            "Could not load image: "
            + str(example["image_path"])
        )

    x1, y1, x2, y2 = [
        int(round(value))
        for value
        in example["bbox"]
    ]

    # Draw YOLO prediction.
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        BOX_COLOR,
        3,
    )

    label = (
        f"{example['class_name']} "
        f"{example['confidence']:.3f}"
    )

    cv2.putText(
        image,
        label,
        (
            max(10, x1),
            max(25, y1 - 10),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        BOX_COLOR,
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Preserve aspect ratio and fit inside panel.
    # --------------------------------------------------------

    h, w = image.shape[:2]

    scale = min(
        PANEL_WIDTH / w,
        PANEL_HEIGHT / h,
    )

    new_width = int(w * scale)
    new_height = int(h * scale)

    resized = cv2.resize(
        image,
        (
            new_width,
            new_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros(
        (
            PANEL_HEIGHT,
            PANEL_WIDTH,
            3,
        ),
        dtype=np.uint8,
    )

    x_offset = (
        PANEL_WIDTH - new_width
    ) // 2

    y_offset = (
        PANEL_HEIGHT - new_height
    ) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width,
    ] = resized

    # --------------------------------------------------------
    # Add white information header.
    # --------------------------------------------------------

    panel = cv2.copyMakeBorder(
        canvas,
        HEADER_HEIGHT,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    actual_percentage = (
        example["confidence"]
        * 100
    )

    target_percentage = (
        target_confidence
        * 100
    )

    cv2.putText(
        panel,
        example["class_name"],
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (35, 35, 35),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        panel,
        (
            f"Target: "
            f"{target_percentage:.0f}%"
            f" | Actual: "
            f"{actual_percentage:.2f}%"
        ),
        (15, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (45, 45, 45),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        panel,
        (
            f"IoU: "
            f"{example['iou']:.3f}"
        ),
        (15, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (45, 45, 45),
        1,
        cv2.LINE_AA,
    )

    # IMAGE NAME IS PRINTED HERE.
    cv2.putText(
        panel,
        (
            f"Image: "
            f"{example['image_path'].name}"
        ),
        (15, 106),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (45, 45, 45),
        1,
        cv2.LINE_AA,
    )

    return panel


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(selected):

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "Tumor Class",
                "Target Confidence (%)",
                "Actual Confidence (%)",
                "IoU",
                "Image Filename",
            ]
        )

        for class_id in TUMOR_CLASSES:

            for target in (
                TARGET_CONFIDENCES
            ):

                example = (
                    selected[
                        class_id
                    ][target]
                )

                if example is None:
                    continue

                writer.writerow(
                    [
                        example[
                            "class_name"
                        ],
                        round(
                            target * 100,
                            2,
                        ),
                        round(
                            example[
                                "confidence"
                            ] * 100,
                            2,
                        ),
                        round(
                            example["iou"],
                            4,
                        ),
                        example[
                            "image_path"
                        ].name,
                    ]
                )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "YOLO26 Confidence-Level "
        "Tumor Detection Examples"
    )
    print("=" * 75)

    if not VAL_IMAGES.exists():
        raise FileNotFoundError(
            f"Validation images not found:\n"
            f"{VAL_IMAGES}"
        )

    if not VAL_LABELS.exists():
        raise FileNotFoundError(
            f"Validation labels not found:\n"
            f"{VAL_LABELS}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO26 model not found:\n"
            f"{MODEL_PATH}"
        )

    print(
        f"Model:\n{MODEL_PATH}"
    )

    print(
        "\nLoading YOLO26..."
    )

    model = YOLO(
        str(MODEL_PATH)
    )

    selected = find_examples(
        model
    )

    # --------------------------------------------------------
    # Print selected examples.
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("SELECTED EXAMPLES")
    print("=" * 75)

    for class_id, class_name in (
        TUMOR_CLASSES.items()
    ):

        print(
            f"\n{class_name}"
        )

        for target in TARGET_CONFIDENCES:

            example = (
                selected[
                    class_id
                ][target]
            )

            if example is None:

                print(
                    f"  Target "
                    f"{target * 100:.0f}%: "
                    "No matching detection."
                )

                continue

            difference = abs(
                example["confidence"]
                - target
            )

            print(
                f"  Target "
                f"{target * 100:.0f}%"
            )

            print(
                f"    Image      : "
                f"{example['image_path'].name}"
            )

            print(
                f"    Confidence : "
                f"{example['confidence'] * 100:.2f}%"
            )

            print(
                f"    IoU        : "
                f"{example['iou']:.4f}"
            )

            # Warn when the closest result
            # is still far from target.
            if difference > 0.05:

                print(
                    "    WARNING: "
                    "closest available result "
                    "is more than 5 percentage "
                    "points from target."
                )

    # --------------------------------------------------------
    # Create 3x3 figure
    #
    # Rows:
    #   Glioma
    #   Meningioma
    #   Pituitary
    #
    # Columns:
    #   ~85%
    #   ~90%
    #   ~95%
    # --------------------------------------------------------

    rows = []

    for class_id in TUMOR_CLASSES:

        panels = []

        for target in TARGET_CONFIDENCES:

            example = (
                selected[
                    class_id
                ][target]
            )

            if example is None:

                empty = np.full(
                    (
                        PANEL_HEIGHT
                        + HEADER_HEIGHT,
                        PANEL_WIDTH,
                        3,
                    ),
                    255,
                    dtype=np.uint8,
                )

                cv2.putText(
                    empty,
                    "No valid detection",
                    (60, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 200),
                    2,
                    cv2.LINE_AA,
                )

                panels.append(
                    empty
                )

            else:

                panels.append(
                    create_panel(
                        example,
                        target,
                    )
                )

        # Light separators.
        spacer = np.full(
            (
                panels[0].shape[0],
                8,
                3,
            ),
            235,
            dtype=np.uint8,
        )

        row = cv2.hconcat(
            [
                panels[0],
                spacer,
                panels[1],
                spacer,
                panels[2],
            ]
        )

        rows.append(row)

    horizontal_separator = np.full(
        (
            8,
            rows[0].shape[1],
            3,
        ),
        235,
        dtype=np.uint8,
    )

    final_figure = cv2.vconcat(
        [
            rows[0],
            horizontal_separator,
            rows[1],
            horizontal_separator,
            rows[2],
        ]
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    OUTPUT_IMAGE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(OUTPUT_IMAGE),
        final_figure,
    )

    save_csv(
        selected
    )

    print("\n")
    print("=" * 75)
    print(
        "FIGURE GENERATED SUCCESSFULLY"
    )
    print("=" * 75)

    print(
        f"\nFigure saved to:\n"
        f"{OUTPUT_IMAGE}"
    )

    print(
        f"\nCSV saved to:\n"
        f"{OUTPUT_CSV}"
    )

    print("=" * 75)

    cv2.imshow(
        "YOLO26 Confidence Examples",
        final_figure,
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()