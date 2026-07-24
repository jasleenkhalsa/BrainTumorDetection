"""
YOLOv8 Configuration File

This file stores every configurable parameter used for training,
validation and inference.

"""

from pathlib import Path
import torch

# -------------------------------------------------------
# Project Paths
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "dataset" / "processed"

DATA_YAML = PROJECT_ROOT / "dataset" / "data.yaml"

RUNS_DIR = PROJECT_ROOT / "runs"

WEIGHTS_DIR = PROJECT_ROOT / "weights"

# -------------------------------------------------------
# Model
# -------------------------------------------------------

MODEL_NAME = "yolov8s.pt"

# -------------------------------------------------------
# Training
# -------------------------------------------------------

IMAGE_SIZE = 640

EPOCHS = 100

BATCH_SIZE = 4

WORKERS = 0

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------
# Optimizer
# -------------------------------------------------------

LEARNING_RATE = 0.001

PATIENCE = 20

# -------------------------------------------------------
# Save
# -------------------------------------------------------

PROJECT_NAME = "BrainTumor"

EXPERIMENT_NAME = "YOLOv8_Baseline"