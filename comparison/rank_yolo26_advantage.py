from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


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

YOLO26_MODEL = (
    PROJECT_ROOT
    / "runs"
    / "YOLO26_Baseline"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = PROJECT_ROOT / "comparison" / "ranked_results"
CSV_PATH = OUTPUT_DIR / "yolo26_vs_yolov8_ranked.csv"

CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640
SMALL_AREA_THRESHOLD = 0.02
TOP_EXAMPLES = 10


def read_yolo_labels(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []

    if not label_path.exists():
        return boxes

    for line in label_path.read_text(
        encoding="utf-8"
    ).splitlines():
        parts = line.strip().split()

        if len(parts) != 5:
            continue

        class_id = int(float(parts[0]))
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        x1 = (x_center - width / 2) * image_width
        y1 = (y_center - height / 2) * image_height
        x2 = (x_center + width / 2) * image_width
        y2 = (y_center + height / 2) * image_height

        boxes.append({
            "class_id": class_id,
            "bbox": [x1, y1, x2, y2],
            "relative_area": width * height,
        })

    return boxes


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

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(
        0.0,
        ay2 - ay1,
    )

    area_b = max(0.0, bx2 - bx1) * max(
        0.0,
        by2 - by1,
    )

    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def best_match_for_ground_truth(
    result,
    ground_truth: dict[str, Any],
) -> dict[str, float] | None:
    if result.boxes is None or len(result.boxes) == 0:
        return None

    best_match = None
    best_iou = 0.0

    for box in result.boxes:
        predicted_class = int(box.cls.item())

        if predicted_class != ground_truth["class_id"]:
            continue

        predicted_bbox = [
            float(value)
            for value in box.xyxy[0].tolist()
        ]

        iou = calculate_iou(
            predicted_bbox,
            ground_truth["bbox"],
        )

        if iou > best_iou:
            best_iou = iou

            best_match = {
                "iou": iou,
                "confidence": float(box.conf.item()),
            }

    return best_match


def add_heading(
    image,
    heading: str,
    color: tuple[int, int, int],
):
    image_with_heading = cv2.copyMakeBorder(
        image,
        55,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    cv2.putText(
        image_with_heading,
        heading,
        (15, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        color,
        2,
        cv2.LINE_AA,
    )

    return image_with_heading


def resize_to_height(image, height: int):
    current_height, current_width = image.shape[:2]

    width = int(
        current_width * height / current_height
    )

    return cv2.resize(
        image,
        (width, height),
    )


def save_comparison(
    image_path: Path,
    yolov8_result,
    yolov26_result,
    row: dict[str, Any],
    output_path: Path,
) -> None:
    original = cv2.imread(str(image_path))

    if original is None:
        return

    original = add_heading(
        original,
        (
            f"Ground Truth | Small area="
            f"{row['relative_area_percent']:.2f}%"
        ),
        (100, 80, 0),
    )

    yolov8_image = add_heading(
        yolov8_result.plot(),
        (
            f"YOLOv8 | IoU={row['yolov8_iou']:.3f} | "
            f"Conf={row['yolov8_confidence']:.3f}"
        ),
        (0, 0, 255),
    )

    yolov26_image = add_heading(
        yolov26_result.plot(),
        (
            f"YOLO26 | IoU={row['yolo26_iou']:.3f} | "
            f"Conf={row['yolo26_confidence']:.3f}"
        ),
        (0, 140, 0),
    )

    common_height = min(
        original.shape[0],
        yolov8_image.shape[0],
        yolov26_image.shape[0],
    )

    original = resize_to_height(
        original,
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
        original,
        yolov8_image,
        yolov26_image,
    ])

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        comparison,
    )


def main() -> None:
    for path in [
        VAL_IMAGES,
        VAL_LABELS,
        YOLOV8_MODEL,
        YOLO26_MODEL,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required path not found:\n{path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("Ranking cases where YOLO26 outperforms YOLOv8")
    print("=" * 70)

    yolov8 = YOLO(str(YOLOV8_MODEL))
    yolov26 = YOLO(str(YOLO26_MODEL))

    image_paths = sorted([
        *VAL_IMAGES.glob("*.jpg"),
        *VAL_IMAGES.glob("*.jpeg"),
        *VAL_IMAGES.glob("*.png"),
    ])

    rows: list[dict[str, Any]] = []

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        image = cv2.imread(str(image_path))

        if image is None:
            continue

        image_height, image_width = image.shape[:2]

        label_path = (
            VAL_LABELS
            / f"{image_path.stem}.txt"
        )

        ground_truth_boxes = read_yolo_labels(
            label_path,
            image_width,
            image_height,
        )

        small_boxes = [
            box
            for box in ground_truth_boxes
            if box["relative_area"] < SMALL_AREA_THRESHOLD
        ]

        if not small_boxes:
            continue

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

        for ground_truth_index, ground_truth in enumerate(
            small_boxes,
            start=1,
        ):
            yolov8_match = best_match_for_ground_truth(
                yolov8_result,
                ground_truth,
            )

            yolov26_match = best_match_for_ground_truth(
                yolov26_result,
                ground_truth,
            )

            yolov8_iou = (
                yolov8_match["iou"]
                if yolov8_match
                else 0.0
            )

            yolov8_confidence = (
                yolov8_match["confidence"]
                if yolov8_match
                else 0.0
            )

            yolo26_iou = (
                yolov26_match["iou"]
                if yolov26_match
                else 0.0
            )

            yolo26_confidence = (
                yolov26_match["confidence"]
                if yolov26_match
                else 0.0
            )

            rows.append({
                "image_name": image_path.name,
                "image_path": str(image_path),
                "ground_truth_index": ground_truth_index,
                "class_id": ground_truth["class_id"],
                "relative_area_percent": (
                    ground_truth["relative_area"] * 100
                ),
                "yolov8_iou": yolov8_iou,
                "yolov8_confidence": yolov8_confidence,
                "yolo26_iou": yolo26_iou,
                "yolo26_confidence": yolo26_confidence,
                "iou_advantage": yolo26_iou - yolov8_iou,
                "confidence_advantage": (
                    yolo26_confidence
                    - yolov8_confidence
                ),
            })

        if index % 50 == 0:
            print(
                f"Processed {index}/{len(image_paths)} images..."
            )

    rows.sort(
        key=lambda row: (
            row["iou_advantage"],
            row["confidence_advantage"],
        ),
        reverse=True,
    )

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image_name",
                "ground_truth_index",
                "class_id",
                "relative_area_percent",
                "yolov8_iou",
                "yolov8_confidence",
                "yolo26_iou",
                "yolo26_confidence",
                "iou_advantage",
                "confidence_advantage",
                "image_path",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV saved to:\n{CSV_PATH}")

    top_rows = rows[:TOP_EXAMPLES]

    for rank, row in enumerate(
        top_rows,
        start=1,
    ):
        image_path = Path(row["image_path"])

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

        output_path = (
            OUTPUT_DIR
            / (
                f"{rank:02d}_"
                f"{image_path.stem}_comparison.jpg"
            )
        )

        save_comparison(
            image_path=image_path,
            yolov8_result=yolov8_result,
            yolov26_result=yolov26_result,
            row=row,
            output_path=output_path,
        )

        print(
            f"\nRank {rank}: {image_path.name}"
        )
        print(
            f"YOLOv8 IoU: {row['yolov8_iou']:.4f}"
        )
        print(
            f"YOLO26 IoU: {row['yolo26_iou']:.4f}"
        )
        print(
            f"IoU gain:   {row['iou_advantage']:.4f}"
        )
        print(
            f"Saved: {output_path}"
        )

    print("\nSearch complete.")


if __name__ == "__main__":
    main()