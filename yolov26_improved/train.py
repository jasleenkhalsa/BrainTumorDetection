from pathlib import Path
import time

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "dataset" / "data.yaml"
RUNS_PATH = PROJECT_ROOT / "runs"

EXPERIMENT_NAME = "YOLO26_Improved"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found:\n{DATA_PATH}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check the active virtual environment."
        )

    print("=" * 70)
    print("Improved YOLO26n Brain Tumor Training")
    print("=" * 70)
    print(f"Dataset: {DATA_PATH}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")

    model = YOLO("yolo26n.pt")

    start_time = time.time()

    model.train(
        data=str(DATA_PATH),

        # Training duration
        epochs=150,
        patience=30,

        # Input resolution
        imgsz=768,

        # Hardware-safe settings
        batch=1,
        workers=0,
        device=0,
        cache=False,
        amp=True,

        # Optimisation
        optimizer="auto",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3.0,

        # MRI-safe augmentation
        degrees=5.0,
        translate=0.05,
        scale=0.20,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,

        # Reduce aggressive augmentation
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.10,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=15,

        # Reproducibility
        seed=42,
        deterministic=True,

        # Saving
        project=str(RUNS_PATH),
        name=EXPERIMENT_NAME,
        exist_ok=False,
        save=True,
        save_period=1,
        plots=True,
        verbose=True,
    )

    elapsed_minutes = (time.time() - start_time) / 60

    print("\n" + "=" * 70)
    print("Improved YOLO26 training completed")
    print(f"Training time: {elapsed_minutes:.2f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()