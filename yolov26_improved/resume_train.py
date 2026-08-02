from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "runs"
    / "YOLO26_Improved-2"
    / "weights"
    / "last.pt"
)


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Restart Windows and verify PyTorch CUDA."
        )

    print("=" * 65)
    print("Resuming Improved YOLO26 Training")
    print("=" * 65)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    model = YOLO(str(CHECKPOINT_PATH))

    model.train(
        resume=True,
        device=0,
        workers=0,
        batch=1,
        cache=False,
        save_period=1,
    )


if __name__ == "__main__":
    main()