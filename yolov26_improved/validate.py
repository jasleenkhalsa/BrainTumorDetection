from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "YOLO26_Improved"
    / "weights"
    / "best.pt"
)

DATA_PATH = PROJECT_ROOT / "dataset" / "data.yaml"
RUNS_PATH = PROJECT_ROOT / "runs"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found:\n{DATA_PATH}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data=str(DATA_PATH),
        imgsz=768,
        batch=1,
        workers=0,
        device=0,
        project=str(RUNS_PATH),
        name="YOLO26_Improved_Validation",
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    print("\n" + "-" * 55)
    print("Improved YOLO26 validation completed")
    print("-" * 55)
    print(f"Precision:       {metrics.box.mp:.4f}")
    print(f"Recall:          {metrics.box.mr:.4f}")
    print(f"mAP@0.5:         {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95:    {metrics.box.map:.4f}")
    print("-" * 55)


if __name__ == "__main__":
    main()