from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "runs"
    / "YOLO26_Baseline"
    / "weights"
    / "last.pt"
)


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"YOLO26 checkpoint was not found:\n{CHECKPOINT_PATH}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Restart the laptop and check nvidia-smi."
        )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Resuming from: {CHECKPOINT_PATH}")

    model = YOLO(str(CHECKPOINT_PATH))
    model.train(resume=True)


if __name__ == "__main__":
    main()