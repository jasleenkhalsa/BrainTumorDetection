from pathlib import Path
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = PROJECT_ROOT / "runs" / "YOLO12_Baseline" / "weights" / "last.pt"


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    print(f"Resuming from: {CHECKPOINT}")

    model = YOLO(str(CHECKPOINT))
    model.train(resume=True)


if __name__ == "__main__":
    main()