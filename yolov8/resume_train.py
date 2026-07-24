from ultralytics import YOLO
from pathlib import Path


# Change this path to your actual last.pt location
CHECKPOINT_PATH = Path(
    r"D:\BrainTumorDetection\runs\YOLOv8_Baseline\weights\last.pt"
)


def resume_training():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    print(f"Resuming from:\n{CHECKPOINT_PATH}")

    model = YOLO(str(CHECKPOINT_PATH))
    model.train(resume=True)


if __name__ == "__main__":
    resume_training()