from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "YOLO26_Baseline"
    / "weights"
    / "best.pt"
)

DATA_PATH = PROJECT_ROOT / "dataset" / "data.yaml"
RUNS_PATH = PROJECT_ROOT / "runs"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO26 best model was not found:\n{MODEL_PATH}"
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset configuration file was not found:\n{DATA_PATH}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Restart the laptop and check nvidia-smi."
        )

    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {DATA_PATH}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data=str(DATA_PATH),
        imgsz=640,
        batch=1,
        workers=0,
        device=0,
        project=str(RUNS_PATH),
        name="YOLO26_Validation",
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    print("\nYOLO26 validation completed successfully.")
    print("-" * 50)
    print(f"Precision:       {metrics.box.mp:.4f}")
    print(f"Recall:          {metrics.box.mr:.4f}")
    print(f"mAP@0.5:         {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95:    {metrics.box.map:.4f}")
    print("-" * 50)
    print(f"Results saved in: {RUNS_PATH / 'YOLO26_Validation'}")


if __name__ == "__main__":
    main()