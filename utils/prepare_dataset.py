"""
Project: Brain Tumor Detection using YOLO

Description:
Converts the BTY11 dataset into the standard YOLO folder structure.
"""

from pathlib import Path
import shutil

# ----------------------------
# Project Paths
# ----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATASET = PROJECT_ROOT / "dataset" / "raw" / "BTY11"

PROCESSED_DATASET = PROJECT_ROOT / "dataset" / "processed"

TRAIN_OUTPUT = PROCESSED_DATASET / "train"
VAL_OUTPUT = PROCESSED_DATASET / "val"

CLASSES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

def create_folders():
    """
    Creates the processed dataset folders.
    """

    folders = [
        TRAIN_OUTPUT / "images",
        TRAIN_OUTPUT / "labels",
        VAL_OUTPUT / "images",
        VAL_OUTPUT / "labels"
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

    print("Folders created successfully.")


def copy_dataset(split_name, destination):

    split_path = RAW_DATASET / split_name

    total_images = 0

    for tumor_class in CLASSES:

        class_folder = split_path / tumor_class

        image_folder = class_folder / "images"

        label_folder = class_folder / "labels"

        if not image_folder.exists():
            print(f"Missing image folder: {image_folder}")
            continue

        for image_path in image_folder.iterdir():

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            # Try exact filename first
            label_path = label_folder / f"{image_path.stem}.txt"

# If not found, try removing spaces
            if not label_path.exists():
                alternative_name = image_path.stem.replace("(", " (")
                alternative_path = label_folder / f"{alternative_name}.txt"

                if alternative_path.exists():
                    label_path = alternative_path
                else:
                    print(f"Missing label for {image_path.name}")
                    continue

            shutil.copy2(
                image_path,
                destination / "images" / image_path.name
            )

            shutil.copy2(
                label_path,
                destination / "labels" / label_path.name
            )

            total_images += 1

    print(f"{split_name}: {total_images} images copied.")


def main():

    print("=" * 50)
    print("Preparing BTY11 Dataset")
    print("=" * 50)

    create_folders()

    copy_dataset("Train", TRAIN_OUTPUT)

    copy_dataset("Val", VAL_OUTPUT)

    print("\nDataset preparation completed successfully.")


if __name__ == "__main__":
    main()