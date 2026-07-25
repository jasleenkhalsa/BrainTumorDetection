from pathlib import Path
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "runs" / "YOLO12_Baseline" / "weights" / "best.pt"
DATA_PATH = PROJECT_ROOT / "dataset" / "data.yaml"
RUNS_PATH = PROJECT_ROOT / "runs"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data=str(DATA_PATH),
        imgsz=640,
        batch=4,
        workers=0,
        device=0,
        project=str(RUNS_PATH),
        name="YOLO12_Validation"
    )

    print("\nValidation completed successfully.")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()