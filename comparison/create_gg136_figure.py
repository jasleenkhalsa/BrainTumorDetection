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

LABEL_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "val"
    / "labels"
    / "gg (136).txt"
)

YOLOV8_MODEL = (
    PROJECT_ROOT
    / "runs"
    / "YOLOv8_Baseline"
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
    / "gg136_small_tumor_comparison.jpg"
)

CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 640

# The fourth ground-truth annotation is the tiny lesion.
TARGET_GROUND_TRUTH_INDEX = 4


def read_target_box(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    lines = [
        line.strip()
        for line in label_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(lines) < TARGET_GROUND_TRUTH_INDEX:
        raise ValueError(
            "The requested ground-truth annotation does not exist."
        )

    target_line = lines[TARGET_GROUND_TRUTH_INDEX - 1]
    parts = target_line.split()

    if len(parts) != 5:
        raise ValueError(
            f"Invalid YOLO label: {target_line}"
        )

    _, x_center, y_center, width, height = map(
        float,
        parts,
    )

    x1 = int(
        round(
            (x_center - width / 2)
            * image_width
        )
    )

    y1 = int(
        round(
            (y_center - height / 2)
            * image_height
        )
    )

    x2 = int(
        round(
            (x_center + width / 2)
            * image_width
        )
    )

    y2 = int(
        round(
            (y_center + height / 2)
            * image_height
        )
    )

    return x1, y1, x2, y2


def draw_target_ground_truth(
    image,
    box: tuple[int, int, int, int],
):
    output = image.copy()

    x1, y1, x2, y2 = box

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (0, 255, 255),
        3,
    )

    cv2.putText(
        output,
        "Tiny Glioma Ground Truth",
        (max(x1 - 20, 5), max(y1 - 12, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


def add_title(
    image,
    title: str,
    color: tuple[int, int, int],
):
    title_height = 60

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
        title,
        (15, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.76,
        color,
        2,
        cv2.LINE_AA,
    )

    return canvas


def resize_to_height(image, target_height: int):
    height, width = image.shape[:2]

    new_width = int(
        width * target_height / height
    )

    return cv2.resize(
        image,
        (new_width, target_height),
    )


def main() -> None:
    for path in [
        IMAGE_PATH,
        LABEL_PATH,
        YOLOV8_MODEL,
        YOLO26_MODEL,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    original = cv2.imread(str(IMAGE_PATH))

    if original is None:
        raise ValueError(
            f"Could not read image:\n{IMAGE_PATH}"
        )

    image_height, image_width = original.shape[:2]

    target_box = read_target_box(
        label_path=LABEL_PATH,
        image_width=image_width,
        image_height=image_height,
    )

    print(f"Image dimensions: {image_width} x {image_height}")
    print(f"Target box: {target_box}")

    yolov8 = YOLO(str(YOLOV8_MODEL))
    yolov26 = YOLO(str(YOLO26_MODEL))

    yolov8_result = yolov8.predict(
        source=str(IMAGE_PATH),
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        iou=0.45,
        device=0,
        verbose=False,
    )[0]

    yolov26_result = yolov26.predict(
        source=str(IMAGE_PATH),
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        iou=0.45,
        device=0,
        verbose=False,
    )[0]

    ground_truth_panel = draw_target_ground_truth(
        original,
        target_box,
    )

    yolov8_panel = yolov8_result.plot()
    yolov26_panel = yolov26_result.plot()

    ground_truth_panel = add_title(
        ground_truth_panel,
        "Ground Truth | Glioma | Area = 0.0797%",
        (130, 100, 0),
    )

    yolov8_panel = add_title(
        yolov8_panel,
        "YOLOv8 | No Correct Match | IoU = 0.000",
        (0, 0, 255),
    )

    yolov26_panel = add_title(
        yolov26_panel,
        "YOLO26 | Detected | Conf = 0.578 | IoU = 0.684",
        (0, 140, 0),
    )

    common_height = min(
        ground_truth_panel.shape[0],
        yolov8_panel.shape[0],
        yolov26_panel.shape[0],
    )

    ground_truth_panel = resize_to_height(
        ground_truth_panel,
        common_height,
    )

    yolov8_panel = resize_to_height(
        yolov8_panel,
        common_height,
    )

    yolov26_panel = resize_to_height(
        yolov26_panel,
        common_height,
    )

    comparison = cv2.hconcat([
        ground_truth_panel,
        yolov8_panel,
        yolov26_panel,
    ])

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(OUTPUT_PATH),
        comparison,
    ):
        raise RuntimeError(
            "Could not save comparison figure."
        )

    print("\nComparison created successfully.")
    print(f"Saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()