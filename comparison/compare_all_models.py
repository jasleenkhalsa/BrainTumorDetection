from __future__ import annotations

import csv
import time
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "dataset" / "data.yaml"
RUNS_PATH = PROJECT_ROOT / "runs"

MODELS = {
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

OUTPUT_CSV = (
    PROJECT_ROOT
    / "comparison"
    / "model_metrics_comparison.csv"
)

IMAGE_SIZE = 640
BATCH_SIZE = 1
DEVICE = 0
WORKERS = 0


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found:\n{DATA_PATH}"
        )

    for model_name, model_path in MODELS.items():
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_name} model not found:\n{model_path}"
            )

    rows = []

    for model_name, model_path in MODELS.items():
        print("\n" + "=" * 70)
        print(f"Validating {model_name}")
        print("=" * 70)
        print(f"Model: {model_path}")

        model = YOLO(str(model_path))

        start_time = time.time()

        metrics = model.val(
            data=str(DATA_PATH),
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            workers=WORKERS,
            plots=True,
            verbose=True,
            project=str(RUNS_PATH),
            name=f"{model_name}_Common_Validation",
            exist_ok=True,
        )

        elapsed_seconds = time.time() - start_time

        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)
        map50 = float(metrics.box.map50)
        map50_95 = float(metrics.box.map)

        speed = getattr(metrics, "speed", {}) or {}

        preprocess_ms = float(
            speed.get("preprocess", 0)
        )

        inference_ms = float(
            speed.get("inference", 0)
        )

        postprocess_ms = float(
            speed.get("postprocess", 0)
        )

        fps = (
            1000 / inference_ms
            if inference_ms > 0
            else 0
        )

        rows.append({
            "model": model_name,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "map50": round(map50, 4),
            "map50_95": round(map50_95, 4),
            "preprocess_ms": round(preprocess_ms, 3),
            "inference_ms": round(inference_ms, 3),
            "postprocess_ms": round(postprocess_ms, 3),
            "fps": round(fps, 2),
            "total_validation_seconds": round(
                elapsed_seconds,
                2,
            ),
        })

        print(f"\n{model_name} completed")
        print(f"Precision      : {precision:.4f}")
        print(f"Recall         : {recall:.4f}")
        print(f"mAP@0.5        : {map50:.4f}")
        print(f"mAP@0.5:0.95   : {map50_95:.4f}")
        print(f"Inference time : {inference_ms:.3f} ms")
        print(f"FPS            : {fps:.2f}")

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "model",
                "precision",
                "recall",
                "map50",
                "map50_95",
                "preprocess_ms",
                "inference_ms",
                "postprocess_ms",
                "fps",
                "total_validation_seconds",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 70)
    print("Final comparison")
    print("=" * 70)

    for row in rows:
        print(
            f"{row['model']}: "
            f"P={row['precision']}, "
            f"R={row['recall']}, "
            f"mAP50={row['map50']}, "
            f"mAP50-95={row['map50_95']}, "
            f"FPS={row['fps']}"
        )

    print(f"\nCSV saved to:\n{OUTPUT_CSV}")


if __name__ == "__main__":
    main()