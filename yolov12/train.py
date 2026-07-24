from pathlib import Path
from ultralytics import YOLO
import os


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "dataset" / "data.yaml"
PROJECT_DIR = PROJECT_ROOT / "runs"


def main():
    print("Current Working Directory:", os.getcwd())
    print("Dataset path:", DATA_PATH)
    print("Results path:", PROJECT_DIR / "YOLO12_Baseline")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATA_PATH}"
        )

    model = YOLO("yolo12s.pt")

    model.train(
        data=str(DATA_PATH),
        epochs=100,
        imgsz=640,
        batch=4,
        workers=0,
        device=0,
        lr0=0.001,
        patience=20,
        amp=True,
        cache=False,

        project=str(PROJECT_DIR),
        name="YOLO12_Baseline",

        save=True,
        save_period=5,
        exist_ok=False
    )


if __name__ == "__main__":
    main()