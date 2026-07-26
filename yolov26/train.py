from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "dataset" / "data.yaml"
RUNS_PATH = PROJECT_ROOT / "runs"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset configuration file was not found:\n{DATA_PATH}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Check the NVIDIA driver and PyTorch installation."
        )

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset file: {DATA_PATH}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Official pretrained YOLO26 nano detection model.
    # It will download automatically if it is not already present.
    model = YOLO("yolo26n.pt")

    model.train(
        data=str(DATA_PATH),

        # Baseline training configuration
        epochs=100,
        imgsz=640,
        batch=2,
        workers=0,
        device=0,

        # Optimisation
        lr0=0.001,
        patience=20,
        amp=True,

        # System stability
        cache=False,
        save=True,
        save_period=2,

        # Output location
        project=str(RUNS_PATH),
        name="YOLO26_Baseline",
        exist_ok=False,

        # Reproducibility
        seed=42,
        deterministic=True,

        # Training plots and logs
        plots=True,
        verbose=True,
    )

    print("\nYOLO26 baseline training completed successfully.")


if __name__ == "__main__":
    main()