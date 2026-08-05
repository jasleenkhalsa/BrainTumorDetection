from __future__ import annotations

from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "val"
    / "images"
    / "gg (136).jpg"
)

YOLOV8_MODEL = (
    PROJECT_ROOT
    / "runs"
    / "YOLOv8_Baseline"
    / "weights"
    / "best.pt"
)

YOLO12_MODEL = (
    PROJECT_ROOT
    / "runs"
    / "YOLO12_Baseline"
    / "weights"
    / "best.pt"
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
    / "gg136_yolov8_yolo12_yolo26_comparison.jpg"
)

CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640


def add_title(image, title: str):
    """Add a white heading above a prediction image."""

    title_height = 60

    panel = cv2.copyMakeBorder(
        image,
        title_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    cv2.putText(
        panel,
        title,
        (15, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    return panel


def resize_to_height(image, target_height: int):
    """Resize an image while preserving its aspect ratio."""

    height, width = image.shape[:2]

    new_width = int(
        width * target_height / height
    )

    return cv2.resize(
        image,
        (new_width, target_height),
    )


def run_prediction(model_path: Path):
    """Load one model and run prediction on the selected MRI."""

    model = YOLO(str(model_path))

    result = model.predict(
        source=str(IMAGE_PATH),
        imgsz=IMAGE_SIZE,
        conf=CONF_THRESHOLD,
        iou=NMS_IOU_THRESHOLD,
        device=0,
        verbose=False,
    )[0]

    return result


def print_detections(model_name: str, result) -> None:
    """Print predicted class, confidence and bounding box."""

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    if result.boxes is None or len(result.boxes) == 0:
        print("No detections.")
        return

    for index, box in enumerate(result.boxes, start=1):
        class_id = int(box.cls.item())
        class_name = result.names[class_id]
        confidence = float(box.conf.item())

        bbox = [
            round(float(value), 2)
            for value in box.xyxy[0].tolist()
        ]

        print(
            f"Detection {index}: "
            f"class={class_name}, "
            f"confidence={confidence:.4f}, "
            f"bbox={bbox}"
        )


def main() -> None:
    required_files = [
        IMAGE_PATH,
        YOLOV8_MODEL,
        YOLO12_MODEL,
        YOLO26_MODEL,
    ]

    for required_file in required_files:
        if not required_file.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{required_file}"
            )

    print("Running YOLOv8...")
    yolov8_result = run_prediction(YOLOV8_MODEL)

    print("Running YOLO12...")
    yolo12_result = run_prediction(YOLO12_MODEL)

    print("Running YOLO26...")
    yolo26_result = run_prediction(YOLO26_MODEL)

    print_detections("YOLOv8", yolov8_result)
    print_detections("YOLO12", yolo12_result)
    print_detections("YOLO26", yolo26_result)

    yolov8_panel = add_title(
        yolov8_result.plot(),
        "YOLOv8",
    )

    yolo12_panel = add_title(
        yolo12_result.plot(),
        "YOLO12",
    )

    yolo26_panel = add_title(
        yolo26_result.plot(),
        "YOLO26",
    )

    common_height = min(
        yolov8_panel.shape[0],
        yolo12_panel.shape[0],
        yolo26_panel.shape[0],
    )

    yolov8_panel = resize_to_height(
        yolov8_panel,
        common_height,
    )

    yolo12_panel = resize_to_height(
        yolo12_panel,
        common_height,
    )

    yolo26_panel = resize_to_height(
        yolo26_panel,
        common_height,
    )

    comparison = cv2.hconcat([
        yolov8_panel,
        yolo12_panel,
        yolo26_panel,
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
            "The comparison image could not be saved."
        )

    print("\nComparison image created successfully.")
    print(f"Saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()