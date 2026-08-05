from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "dataset" / "processed"
VAL_IMAGES = DATASET_ROOT / "val" / "images"
VAL_LABELS = DATASET_ROOT / "val" / "labels"

YOLOV8_MODEL = (
    PROJECT_ROOT
    / "runs"
    / "YOLOv8_Baseline"
    / "weights"
    / "best.pt"
)

# Use the better YOLO26 BASELINE model, not Improved-2.
YOLO26_MODEL = (
    PROJECT_ROOT
    / "runs"
    / "YOLO26_Baseline"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "comparison"
    / "small_tumor_examples"
)


# -------------------------------------------------------------------
# Evaluation settings
# -------------------------------------------------------------------

CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45
MATCH_IOU_THRESHOLD = 0.50
IMAGE_SIZE = 640

# YOLO label width × height is already relative to full image area.
# A box smaller than 2% of image area is considered small.
SMALL_AREA_THRESHOLD = 0.02

MAX_EXAMPLES = 5


# -------------------------------------------------------------------
# Read ground-truth labels
# -------------------------------------------------------------------

def read_yolo_labels(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    """
    Read YOLO-format labels and convert normalized coordinates
    into pixel xyxy coordinates.
    """

    boxes: list[dict[str, Any]] = []

    if not label_path.exists():
        return boxes

    lines = label_path.read_text(
        encoding="utf-8"
    ).splitlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) != 5:
            continue

        try:
            class_id = int(float(parts[0]))
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

        except ValueError:
            continue

        relative_area = width * height

        x1 = (x_center - width / 2) * image_width
        y1 = (y_center - height / 2) * image_height
        x2 = (x_center + width / 2) * image_width
        y2 = (y_center + height / 2) * image_height

        boxes.append({
            "class_id": class_id,
            "bbox": [x1, y1, x2, y2],
            "relative_area": relative_area,
        })

    return boxes


# -------------------------------------------------------------------
# IoU calculation
# -------------------------------------------------------------------

def calculate_iou(
    box_a: list[float],
    box_b: list[float],
) -> float:
    """
    Calculate intersection-over-union between two xyxy boxes.
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0.0,
        intersection_x2 - intersection_x1,
    )

    intersection_height = max(
        0.0,
        intersection_y2 - intersection_y1,
    )

    intersection_area = (
        intersection_width * intersection_height
    )

    area_a = max(0.0, ax2 - ax1) * max(
        0.0,
        ay2 - ay1,
    )

    area_b = max(0.0, bx2 - bx1) * max(
        0.0,
        by2 - by1,
    )

    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


# -------------------------------------------------------------------
# Match prediction to small ground-truth tumor
# -------------------------------------------------------------------

def find_best_match(
    prediction,
    ground_truth_boxes: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Return the best matching prediction when:

    1. Predicted class equals ground-truth class
    2. IoU is at least MATCH_IOU_THRESHOLD
    """

    if (
        prediction.boxes is None
        or len(prediction.boxes) == 0
    ):
        return None

    best_match: dict[str, Any] | None = None
    best_iou = 0.0

    for predicted_box in prediction.boxes:
        predicted_class = int(
            predicted_box.cls.item()
        )

        predicted_confidence = float(
            predicted_box.conf.item()
        )

        predicted_coordinates = [
            float(value)
            for value
            in predicted_box.xyxy[0].tolist()
        ]

        for ground_truth in ground_truth_boxes:
            if (
                predicted_class
                != ground_truth["class_id"]
            ):
                continue

            iou = calculate_iou(
                predicted_coordinates,
                ground_truth["bbox"],
            )

            if (
                iou >= MATCH_IOU_THRESHOLD
                and iou > best_iou
            ):
                best_iou = iou

                best_match = {
                    "class_id": predicted_class,
                    "confidence": predicted_confidence,
                    "bbox": predicted_coordinates,
                    "iou": iou,
                }

    return best_match


# -------------------------------------------------------------------
# Visualization helpers
# -------------------------------------------------------------------

def draw_ground_truth(
    image,
    ground_truth_boxes: list[dict[str, Any]],
    class_names: dict[int, str],
):
    """Draw the small ground-truth tumor boxes."""

    output = image.copy()

    for ground_truth in ground_truth_boxes:
        x1, y1, x2, y2 = [
            int(round(value))
            for value in ground_truth["bbox"]
        ]

        class_id = ground_truth["class_id"]

        class_name = class_names.get(
            class_id,
            f"Class {class_id}",
        )

        area_percent = (
            ground_truth["relative_area"] * 100
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (255, 255, 0),
            2,
        )

        label = (
            f"Ground Truth: {class_name} "
            f"({area_percent:.2f}% image area)"
        )

        cv2.putText(
            output,
            label,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
        )

    return output


def add_heading(
    image,
    heading: str,
    text_color: tuple[int, int, int],
):
    """Add a white title bar above an image."""

    title_height = 55

    canvas = cv2.copyMakeBorder(
        image,
        title_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    cv2.putText(
        canvas,
        heading,
        (15, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        text_color,
        2,
        cv2.LINE_AA,
    )

    return canvas


def resize_to_height(image, target_height: int):
    """Resize an image while preserving its aspect ratio."""

    current_height, current_width = image.shape[:2]

    new_width = int(
        current_width
        * target_height
        / current_height
    )

    return cv2.resize(
        image,
        (new_width, target_height),
    )


def save_comparison(
    image_path: Path,
    small_ground_truth_boxes: list[dict[str, Any]],
    yolov8_result,
    yolov26_result,
    yolov26_match: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save three panels:

    1. Ground truth
    2. YOLOv8 result
    3. YOLO26 result
    """

    original = cv2.imread(str(image_path))

    if original is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    ground_truth_image = draw_ground_truth(
        image=original,
        ground_truth_boxes=small_ground_truth_boxes,
        class_names=yolov26_result.names,
    )

    yolov8_image = yolov8_result.plot()
    yolov26_image = yolov26_result.plot()

    ground_truth_image = add_heading(
        ground_truth_image,
        "Ground Truth: Small Tumor",
        (120, 80, 0),
    )

    yolov8_image = add_heading(
        yolov8_image,
        "YOLOv8: No Correct Match",
        (0, 0, 255),
    )

    yolov26_heading = (
        "YOLO26: Correct Detection | "
        f"Conf={yolov26_match['confidence']:.2f} | "
        f"IoU={yolov26_match['iou']:.2f}"
    )

    yolov26_image = add_heading(
        yolov26_image,
        yolov26_heading,
        (0, 140, 0),
    )

    common_height = min(
        ground_truth_image.shape[0],
        yolov8_image.shape[0],
        yolov26_image.shape[0],
    )

    ground_truth_image = resize_to_height(
        ground_truth_image,
        common_height,
    )

    yolov8_image = resize_to_height(
        yolov8_image,
        common_height,
    )

    yolov26_image = resize_to_height(
        yolov26_image,
        common_height,
    )

    comparison = cv2.hconcat([
        ground_truth_image,
        yolov8_image,
        yolov26_image,
    ])

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(output_path),
        comparison,
    )

    if not success:
        raise RuntimeError(
            f"Could not save comparison: {output_path}"
        )


# -------------------------------------------------------------------
# Main process
# -------------------------------------------------------------------

def main() -> None:
    required_paths = [
        VAL_IMAGES,
        VAL_LABELS,
        YOLOV8_MODEL,
        YOLO26_MODEL,
    ]

    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required path not found:\n"
                f"{required_path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("Small-Tumor Detection Comparison")
    print("=" * 70)
    print(f"YOLOv8 model : {YOLOV8_MODEL}")
    print(f"YOLO26 model : {YOLO26_MODEL}")
    print(
        f"Small threshold: "
        f"{SMALL_AREA_THRESHOLD * 100:.1f}% "
        "of image area"
    )
    print(
        f"Correct detection requires IoU >= "
        f"{MATCH_IOU_THRESHOLD}"
    )
    print("=" * 70)

    print("\nLoading models...")

    yolov8 = YOLO(str(YOLOV8_MODEL))
    yolov26 = YOLO(str(YOLO26_MODEL))

    image_paths = sorted([
        *VAL_IMAGES.glob("*.jpg"),
        *VAL_IMAGES.glob("*.jpeg"),
        *VAL_IMAGES.glob("*.png"),
        *VAL_IMAGES.glob("*.bmp"),
    ])

    if not image_paths:
        raise RuntimeError(
            f"No validation images found in:\n"
            f"{VAL_IMAGES}"
        )

    qualifying_small_tumor_images = 0
    found_examples = 0

    for image_number, image_path in enumerate(
        image_paths,
        start=1,
    ):
        original_image = cv2.imread(
            str(image_path)
        )

        if original_image is None:
            print(
                f"Skipping unreadable image: "
                f"{image_path.name}"
            )
            continue

        image_height, image_width = (
            original_image.shape[:2]
        )

        label_path = (
            VAL_LABELS
            / f"{image_path.stem}.txt"
        )

        ground_truth_boxes = read_yolo_labels(
            label_path=label_path,
            image_width=image_width,
            image_height=image_height,
        )

        small_boxes = [
            box
            for box in ground_truth_boxes
            if (
                box["relative_area"]
                < SMALL_AREA_THRESHOLD
            )
        ]

        if not small_boxes:
            continue

        qualifying_small_tumor_images += 1

        yolov8_result = yolov8.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=CONF_THRESHOLD,
            iou=NMS_IOU_THRESHOLD,
            device=0,
            verbose=False,
        )[0]

        yolov26_result = yolov26.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=CONF_THRESHOLD,
            iou=NMS_IOU_THRESHOLD,
            device=0,
            verbose=False,
        )[0]

        yolov8_match = find_best_match(
            prediction=yolov8_result,
            ground_truth_boxes=small_boxes,
        )

        yolov26_match = find_best_match(
            prediction=yolov26_result,
            ground_truth_boxes=small_boxes,
        )

        # Desired paper example:
        # YOLOv8 has no correct small-tumor match,
        # while YOLO26 has a correct match.
        if (
            yolov8_match is None
            and yolov26_match is not None
        ):
            output_path = (
                OUTPUT_DIR
                / (
                    f"{found_examples + 1:02d}_"
                    f"{image_path.stem}_comparison.jpg"
                )
            )

            save_comparison(
                image_path=image_path,
                small_ground_truth_boxes=small_boxes,
                yolov8_result=yolov8_result,
                yolov26_result=yolov26_result,
                yolov26_match=yolov26_match,
                output_path=output_path,
            )

            print("\nExample found")
            print(f"Image      : {image_path.name}")
            print(
                f"YOLO26 conf: "
                f"{yolov26_match['confidence']:.4f}"
            )
            print(
                f"YOLO26 IoU : "
                f"{yolov26_match['iou']:.4f}"
            )
            print(f"Saved to   : {output_path}")

            found_examples += 1

            if found_examples >= MAX_EXAMPLES:
                break

        if image_number % 50 == 0:
            print(
                f"Processed {image_number}/"
                f"{len(image_paths)} images..."
            )

    print("\n" + "=" * 70)
    print("Search completed")
    print("=" * 70)
    print(
        "Validation images containing at least one "
        f"small ground-truth box: "
        f"{qualifying_small_tumor_images}"
    )
    print(
        "Examples where YOLOv8 missed and "
        f"YOLO26 detected: {found_examples}"
    )

    if found_examples == 0:
        print(
            "\nNo valid example was found under the "
            "current evaluation settings."
        )
        print(
            "Do not claim that YOLO26 detected a small "
            "tumor missed by YOLOv8 unless the script "
            "finds a genuine example."
        )
        print(
            "\nYou may perform a separately reported "
            "threshold analysis using conf=0.20, but both "
            "models must use the same threshold."
        )
    else:
        print(
            f"\nComparison figures saved in:\n"
            f"{OUTPUT_DIR}"
        )


if __name__ == "__main__":
    main()