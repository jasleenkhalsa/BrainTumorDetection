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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "comparison"
    / "ranked_all_models"
)

CSV_PATH = (
    OUTPUT_DIR
    / "yolov8_yolo12_yolo26_ranked.csv"
)

CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45
MATCH_IOU_THRESHOLD = 0.50
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


def draw_target_ground_truth(
    image,
    ground_truth: dict[str, Any],
    class_name: str,
):
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

    area_percent = (
        ground_truth["relative_area"] * 100
    )

    label = (
        f"Ground Truth | {class_name} | "
        f"Area={area_percent:.4f}%"
    )

    cv2.putText(
        output,
        label,
        (max(x1 - 40, 10), max(y1 - 12, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
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
    title_height = 65

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
        (15, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
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


def format_model_title(
    model_name: str,
    match: dict[str, float] | None,
) -> str:
    if match is None:
        return (
            f"{model_name} | No Correct Match | "
            "IoU=0.000"
        )

    status = (
        "Detected"
        if match["iou"] >= MATCH_IOU_THRESHOLD
        else "Weak Match"
    )

    return (
        f"{model_name} | {status} | "
        f"Conf={match['confidence']:.3f} | "
        f"IoU={match['iou']:.3f}"
    )


def save_four_panel_comparison(
    image_path: Path,
    ground_truth: dict[str, Any],
    results: dict[str, Any],
    matches: dict[str, dict[str, float] | None],
    output_path: Path,
) -> None:
    original = cv2.imread(str(image_path))

    if original is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    class_id = ground_truth["class_id"]

    class_name = results["YOLO26"].names.get(
        class_id,
        f"Class {class_id}",
    )

    ground_truth_panel = draw_target_ground_truth(
        original,
        ground_truth,
        class_name,
    )

    ground_truth_panel = add_title(
        ground_truth_panel,
        (
            f"Ground Truth | {class_name} | "
            f"Area={ground_truth['relative_area'] * 100:.4f}%"
        ),
        (120, 80, 0),
    )

    yolo8_panel = add_title(
        results["YOLOv8"].plot(),
        format_model_title(
            "YOLOv8",
            matches["YOLOv8"],
        ),
        (0, 0, 255),
    )

    yolo12_panel = add_title(
        results["YOLO12"].plot(),
        format_model_title(
            "YOLO12",
            matches["YOLO12"],
        ),
        (255, 120, 0),
    )

    yolo26_panel = add_title(
        results["YOLO26"].plot(),
        format_model_title(
            "YOLO26",
            matches["YOLO26"],
        ),
        (0, 140, 0),
    )

    common_height = min(
        ground_truth_panel.shape[0],
        yolo8_panel.shape[0],
        yolo12_panel.shape[0],
        yolo26_panel.shape[0],
    )

    panels = [
        resize_to_height(
            ground_truth_panel,
            common_height,
        ),
        resize_to_height(
            yolo8_panel,
            common_height,
        ),
        resize_to_height(
            yolo12_panel,
            common_height,
        ),
        resize_to_height(
            yolo26_panel,
            common_height,
        ),
    ]

    comparison = cv2.hconcat(panels)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(output_path),
        comparison,
    ):
        raise RuntimeError(
            f"Could not save comparison: {output_path}"
        )


def main() -> None:
    required_paths = [
        VAL_IMAGES,
        VAL_LABELS,
        *MODEL_PATHS.values(),
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required path not found:\n{path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 75)
    print("Ranking small-tumor cases for YOLOv8, YOLO12 and YOLO26")
    print("=" * 75)

    models = {
        name: YOLO(str(path))
        for name, path in MODEL_PATHS.items()
    }

    image_paths = sorted([
        *VAL_IMAGES.glob("*.jpg"),
        *VAL_IMAGES.glob("*.jpeg"),
        *VAL_IMAGES.glob("*.png"),
        *VAL_IMAGES.glob("*.bmp"),
    ])

    rows: list[dict[str, Any]] = []

    for image_index, image_path in enumerate(
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

        results = {}

        for model_name, model in models.items():
            results[model_name] = model.predict(
                source=str(image_path),
                imgsz=IMAGE_SIZE,
                conf=CONF_THRESHOLD,
                iou=NMS_IOU_THRESHOLD,
                device=0,
                verbose=False,
            )[0]

        for gt_index, ground_truth in enumerate(
            small_boxes,
            start=1,
        ):
            matches = {
                model_name: best_match_for_ground_truth(
                    result,
                    ground_truth,
                )
                for model_name, result in results.items()
            }

            def get_value(
                model_name: str,
                key: str,
            ) -> float:
                match = matches[model_name]

                if match is None:
                    return 0.0

                return float(match[key])

            yolo8_iou = get_value("YOLOv8", "iou")
            yolo12_iou = get_value("YOLO12", "iou")
            yolo26_iou = get_value("YOLO26", "iou")

            yolo8_conf = get_value(
                "YOLOv8",
                "confidence",
            )

            yolo12_conf = get_value(
                "YOLO12",
                "confidence",
            )

            yolo26_conf = get_value(
                "YOLO26",
                "confidence",
            )

            best_other_iou = max(
                yolo8_iou,
                yolo12_iou,
            )

            yolo26_advantage = (
                yolo26_iou - best_other_iou
            )

            yolo26_only_hit = int(
                yolo26_iou >= MATCH_IOU_THRESHOLD
                and yolo8_iou < MATCH_IOU_THRESHOLD
                and yolo12_iou < MATCH_IOU_THRESHOLD
            )

            yolo26_beats_both = int(
                yolo26_iou > yolo8_iou
                and yolo26_iou > yolo12_iou
            )

            rows.append({
                "image_name": image_path.name,
                "image_path": str(image_path),
                "ground_truth_index": gt_index,
                "class_id": ground_truth["class_id"],
                "relative_area_percent": (
                    ground_truth["relative_area"] * 100
                ),
                "yolov8_iou": yolo8_iou,
                "yolov8_confidence": yolo8_conf,
                "yolo12_iou": yolo12_iou,
                "yolo12_confidence": yolo12_conf,
                "yolo26_iou": yolo26_iou,
                "yolo26_confidence": yolo26_conf,
                "yolo26_advantage_over_best_other": (
                    yolo26_advantage
                ),
                "yolo26_only_hit": yolo26_only_hit,
                "yolo26_beats_both": yolo26_beats_both,
            })

        if image_index % 50 == 0:
            print(
                f"Processed {image_index}/"
                f"{len(image_paths)} images..."
            )

    rows.sort(
        key=lambda row: (
            row["yolo26_only_hit"],
            row["yolo26_beats_both"],
            row[
                "yolo26_advantage_over_best_other"
            ],
            row["yolo26_iou"],
            row["yolo26_confidence"],
        ),
        reverse=True,
    )

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        fieldnames = [
            "image_name",
            "ground_truth_index",
            "class_id",
            "relative_area_percent",
            "yolov8_iou",
            "yolov8_confidence",
            "yolo12_iou",
            "yolo12_confidence",
            "yolo26_iou",
            "yolo26_confidence",
            "yolo26_advantage_over_best_other",
            "yolo26_only_hit",
            "yolo26_beats_both",
            "image_path",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV saved to:\n{CSV_PATH}")

    for rank, row in enumerate(
        rows[:TOP_EXAMPLES],
        start=1,
    ):
        image_path = Path(row["image_path"])

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        height, width = image.shape[:2]

        label_path = (
            VAL_LABELS
            / f"{image_path.stem}.txt"
        )

        ground_truth_boxes = read_yolo_labels(
            label_path,
            width,
            height,
        )

        small_boxes = [
            box
            for box in ground_truth_boxes
            if box["relative_area"] < SMALL_AREA_THRESHOLD
        ]

        ground_truth = small_boxes[
            row["ground_truth_index"] - 1
        ]

        results = {
            model_name: model.predict(
                source=str(image_path),
                imgsz=IMAGE_SIZE,
                conf=CONF_THRESHOLD,
                iou=NMS_IOU_THRESHOLD,
                device=0,
                verbose=False,
            )[0]
            for model_name, model in models.items()
        }

        matches = {
            model_name: best_match_for_ground_truth(
                result,
                ground_truth,
            )
            for model_name, result in results.items()
        }

        output_path = (
            OUTPUT_DIR
            / (
                f"{rank:02d}_"
                f"{image_path.stem}_gt"
                f"{row['ground_truth_index']}_"
                "comparison.jpg"
            )
        )

        save_four_panel_comparison(
            image_path=image_path,
            ground_truth=ground_truth,
            results=results,
            matches=matches,
            output_path=output_path,
        )

        print("\n" + "-" * 65)
        print(f"Rank {rank}: {image_path.name}")
        print(
            f"Ground-truth index: "
            f"{row['ground_truth_index']}"
        )
        print(
            f"Area: "
            f"{row['relative_area_percent']:.4f}%"
        )
        print(
            f"YOLOv8  IoU={row['yolov8_iou']:.4f}, "
            f"Conf={row['yolov8_confidence']:.4f}"
        )
        print(
            f"YOLO12 IoU={row['yolo12_iou']:.4f}, "
            f"Conf={row['yolo12_confidence']:.4f}"
        )
        print(
            f"YOLO26 IoU={row['yolo26_iou']:.4f}, "
            f"Conf={row['yolo26_confidence']:.4f}"
        )
        print(
            "YOLO26 advantage over best other: "
            f"{row['yolo26_advantage_over_best_other']:.4f}"
        )
        print(
            f"YOLO26 only hit: "
            f"{row['yolo26_only_hit']}"
        )
        print(f"Saved: {output_path}")

    print("\n" + "=" * 75)
    print("Ranking completed")
    print("=" * 75)


if __name__ == "__main__":
    main()