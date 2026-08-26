from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VAL_IMAGES = PROJECT_ROOT / "dataset" / "processed" / "val" / "images"
VAL_LABELS = PROJECT_ROOT / "dataset" / "processed" / "val" / "labels"

MODEL_PATHS = {
    "YOLOv8": PROJECT_ROOT / "runs" / "YOLOv8_Baseline" / "weights" / "best.pt",
    "YOLO12": PROJECT_ROOT / "runs" / "YOLO12_Baseline" / "weights" / "best.pt",
    "YOLO26": PROJECT_ROOT / "runs" / "YOLO26_Baseline" / "weights" / "best.pt",
}

OUTPUT_CSV = PROJECT_ROOT / "comparison" / "lesion_size_metrics.csv"


# ============================================================
# SETTINGS
# ============================================================

IMGSZ = 640
CONF_THRESHOLD = 0.25
IOU_MATCH_THRESHOLD = 0.50

CLASS_NAMES = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary",
}

TUMOR_CLASSES = {0, 1, 3}


# ============================================================
# SIZE BINS
# ============================================================

def size_category(relative_area_percent):
    if relative_area_percent < 1.0:
        return "Small (<1%)"
    elif relative_area_percent <= 3.0:
        return "Medium (1-3%)"
    else:
        return "Large (>3%)"


# ============================================================
# LABEL READING
# ============================================================

def read_yolo_labels(label_path):
    annotations = []

    if not label_path.exists():
        return annotations

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
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
# BOX CONVERSION
# ============================================================

def yolo_to_xyxy(annotation, img_w, img_h):
    xc = annotation["x_center"] * img_w
    yc = annotation["y_center"] * img_h
    bw = annotation["width"] * img_w
    bh = annotation["height"] * img_h

    x1 = xc - bw / 2
    y1 = yc - bh / 2
    x2 = xc + bw / 2
    y2 = yc + bh / 2

    return [x1, y1, x2, y2]


# ============================================================
# IOU
# ============================================================

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


# ============================================================
# AP CALCULATION
# ============================================================

def compute_ap(recalls, precisions):
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    indices = np.where(recalls[1:] != recalls[:-1])[0]

    ap = np.sum(
        (recalls[indices + 1] - recalls[indices]) *
        precisions[indices + 1]
    )

    return ap


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(model_name, model):
    print(f"\nEvaluating {model_name}...")

    stats = {
        "Small (<1%)": {
            "gt": 0,
            "tp": 0,
            "fp": 0,
            "pred_records": [],
        },
        "Medium (1-3%)": {
            "gt": 0,
            "tp": 0,
            "fp": 0,
            "pred_records": [],
        },
        "Large (>3%)": {
            "gt": 0,
            "tp": 0,
            "fp": 0,
            "pred_records": [],
        },
    }

    image_paths = sorted(
        list(VAL_IMAGES.glob("*.jpg")) +
        list(VAL_IMAGES.glob("*.jpeg")) +
        list(VAL_IMAGES.glob("*.png"))
    )

    print(f"Validation images: {len(image_paths)}")

    for idx, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))

        if image is None:
            continue

        img_h, img_w = image.shape[:2]

        label_path = VAL_LABELS / f"{image_path.stem}.txt"
        annotations = read_yolo_labels(label_path)

        # ----------------------------------------------------
        # Ground truth tumor lesions
        # ----------------------------------------------------

        gt_objects = []

        for annotation in annotations:
            if annotation["class_id"] not in TUMOR_CLASSES:
                continue

            gt_box = yolo_to_xyxy(annotation, img_w, img_h)

            relative_area = (
                annotation["width"] *
                annotation["height"] *
                100.0
            )

            category = size_category(relative_area)

            gt_objects.append(
                {
                    "class_id": annotation["class_id"],
                    "bbox": gt_box,
                    "category": category,
                    "matched": False,
                }
            )

            stats[category]["gt"] += 1

        # ----------------------------------------------------
        # Model predictions
        # ----------------------------------------------------

        result = model.predict(
            source=str(image_path),
            imgsz=IMGSZ,
            conf=0.001,   # low threshold required for AP calculation
            device=0,
            verbose=False,
        )[0]

        predictions = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())

                if class_id not in TUMOR_CLASSES:
                    continue

                confidence = float(box.conf.item())
                bbox = [float(x) for x in box.xyxy[0].tolist()]

                predictions.append(
                    {
                        "class_id": class_id,
                        "confidence": confidence,
                        "bbox": bbox,
                    }
                )

        # Highest confidence first
        predictions.sort(
            key=lambda x: x["confidence"],
            reverse=True
        )

        # ----------------------------------------------------
        # Match predictions with ground truth
        # ----------------------------------------------------

        for pred in predictions:
            best_iou = 0.0
            best_gt_index = None

            for gt_index, gt in enumerate(gt_objects):
                if gt["matched"]:
                    continue

                if gt["class_id"] != pred["class_id"]:
                    continue

                iou = calculate_iou(
                    pred["bbox"],
                    gt["bbox"],
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = gt_index

            # Correct prediction
            if (
                best_gt_index is not None
                and best_iou >= IOU_MATCH_THRESHOLD
            ):
                gt = gt_objects[best_gt_index]
                gt["matched"] = True

                category = gt["category"]

                stats[category]["pred_records"].append(
                    {
                        "confidence": pred["confidence"],
                        "tp": 1,
                        "fp": 0,
                    }
                )

            else:
                # ------------------------------------------------
                # FP size assignment:
                # assign FP according to predicted box area
                # ------------------------------------------------

                x1, y1, x2, y2 = pred["bbox"]

                box_area = max(0, x2 - x1) * max(0, y2 - y1)
                image_area = img_w * img_h

                if image_area > 0:
                    relative_pred_area = (
                        box_area / image_area
                    ) * 100
                else:
                    relative_pred_area = 0

                category = size_category(relative_pred_area)

                stats[category]["pred_records"].append(
                    {
                        "confidence": pred["confidence"],
                        "tp": 0,
                        "fp": 1,
                    }
                )

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(image_paths)}")

    # ========================================================
    # FINAL METRICS PER SIZE GROUP
    # ========================================================

    results = {}

    for category, values in stats.items():
        gt_count = values["gt"]

        records = sorted(
            values["pred_records"],
            key=lambda x: x["confidence"],
            reverse=True,
        )

        if len(records) == 0:
            results[category] = {
                "recall": 0.0,
                "precision": 0.0,
                "map50": 0.0,
                "gt_count": gt_count,
            }
            continue

        tp_array = np.array(
            [r["tp"] for r in records],
            dtype=float,
        )

        fp_array = np.array(
            [r["fp"] for r in records],
            dtype=float,
        )

        cumulative_tp = np.cumsum(tp_array)
        cumulative_fp = np.cumsum(fp_array)

        recalls = cumulative_tp / max(gt_count, 1)

        precisions = (
            cumulative_tp /
            np.maximum(
                cumulative_tp + cumulative_fp,
                1e-12
            )
        )

        # Recall and precision at chosen confidence threshold
        filtered_records = [
            r for r in records
            if r["confidence"] >= CONF_THRESHOLD
        ]

        tp_at_threshold = sum(
            r["tp"]
            for r in filtered_records
        )

        fp_at_threshold = sum(
            r["fp"]
            for r in filtered_records
        )

        recall_value = (
            tp_at_threshold / gt_count
            if gt_count > 0
            else 0.0
        )

        precision_value = (
            tp_at_threshold /
            (tp_at_threshold + fp_at_threshold)
            if (tp_at_threshold + fp_at_threshold) > 0
            else 0.0
        )

        ap50 = compute_ap(
            recalls,
            precisions,
        )

        results[category] = {
            "recall": recall_value,
            "precision": precision_value,
            "map50": ap50,
            "gt_count": gt_count,
        }

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("Lesion-Size Stratified YOLO Evaluation")
    print("=" * 80)

    models = {}

    for name, path in MODEL_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{name} model not found:\n{path}"
            )

        print(f"\nLoading {name}:")
        print(path)

        models[name] = YOLO(str(path))

    all_results = {}

    for model_name, model in models.items():
        all_results[model_name] = evaluate_model(
            model_name,
            model,
        )

    # ========================================================
    # BUILD FINAL TABLE
    # ========================================================

    rows = []

    categories = [
        "Small (<1%)",
        "Medium (1-3%)",
        "Large (>3%)",
    ]

    for category in categories:
        row = {
            "Lesion size": category,

            "YOLOv8 Recall":
                all_results["YOLOv8"][category]["recall"],

            "YOLO12 Recall":
                all_results["YOLO12"][category]["recall"],

            "YOLO26 Recall":
                all_results["YOLO26"][category]["recall"],

            "YOLO26 Precision":
                all_results["YOLO26"][category]["precision"],

            "YOLO26 mAP@0.5":
                all_results["YOLO26"][category]["map50"],

            "Number of GT lesions":
                all_results["YOLO26"][category]["gt_count"],
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    # Convert to percentages for readability
    percentage_columns = [
        "YOLOv8 Recall",
        "YOLO12 Recall",
        "YOLO26 Recall",
        "YOLO26 Precision",
        "YOLO26 mAP@0.5",
    ]

    for column in percentage_columns:
        df[column] = df[column] * 100

    print("\n")
    print("=" * 80)
    print("FINAL LESION-SIZE RESULTS")
    print("=" * 80)

    print(
        df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print("\nSaved to:")
    print(OUTPUT_CSV)

    print("=" * 80)


if __name__ == "__main__":
    main()