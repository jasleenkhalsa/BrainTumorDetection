"""
YOLOv8 Training Script

Description:
Trains the YOLOv8 model on the Brain Tumor dataset.
"""

from ultralytics import YOLO
import torch
import time

from config import *


def train():
    print("=" * 60)
    print("Brain Tumor Detection using YOLOv8")
    print("=" * 60)

    print(f"PyTorch Version : {torch.__version__}")
    print(f"CUDA Available  : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU : {torch.cuda.get_device_name(0)}")

    model = YOLO(MODEL_NAME)

    start_time = time.time()

    model.train(
        data=str(DATA_YAML),
        imgsz=IMAGE_SIZE,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=DEVICE,
        project=str(RUNS_DIR),
        name=EXPERIMENT_NAME,
        lr0=LEARNING_RATE,
        patience=PATIENCE,
        pretrained=True,
        verbose=True,
        plots=True,
        save=True,
        exist_ok=True,
        seed=42,
        cache=False,
        amp=True
    )

    end_time = time.time()

    print("=" * 60)
    print("Training Completed")
    print("=" * 60)
    print(f"Training Time: {(end_time - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    train()