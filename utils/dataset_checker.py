"""
Project: Brain Tumor Detection using YOLO

Description:
Validates the prepared dataset before training.
"""
# IMPORTS AND PATHS
from pathlib import Path
from PIL import Image
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET = PROJECT_ROOT / "dataset" / "processed"

TRAIN_IMAGES = DATASET / "train" / "images"
TRAIN_LABELS = DATASET / "train" / "labels"

VAL_IMAGES = DATASET / "val" / "images"
VAL_LABELS = DATASET / "val" / "labels"

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

# COUNT IMAGES
def count_files(folder, extensions):

    return len(
        [
            file
            for file in folder.iterdir()
            if file.suffix.lower() in extensions
        ]
    )

# CHECK MISSING LABELS
def check_missing_labels(image_folder, label_folder):

    missing = []

    for image in image_folder.iterdir():

        if image.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        label = label_folder / f"{image.stem}.txt"

        if not label.exists():
            missing.append(image.name)

    return missing

# CHECK CORRUPTED IMAGES
def check_corrupted_images(image_folder):

    corrupted = []

    for image in image_folder.iterdir():

        try:
            Image.open(image).verify()

        except Exception:
            corrupted.append(image.name)

    return corrupted

# VALIDALE LABELS
def validate_labels(label_folder):

    invalid = []

    class_counter = Counter()

    for label in label_folder.glob("*.txt"):

        with open(label, "r") as file:
            lines = file.readlines()

        if len(lines) == 0:
            invalid.append((label.name, "Empty label file"))
            continue

        for line_number, line in enumerate(lines, start=1):

            values = line.strip().split()

            if len(values) != 5:
                invalid.append(
                    (label.name, f"Line {line_number}: Expected 5 values, got {len(values)}")
                )
                continue

            try:
                class_id = int(values[0])

                if class_id >= len(CLASS_NAMES) or class_id < 0:
                    invalid.append(
                        (label.name, f"Line {line_number}: Invalid class id {class_id}")
                    )
                    continue

                class_counter[class_id] += 1

                x, y, w, h = map(float, values[1:])

                if not (
                    0 <= x <= 1 and
                    0 <= y <= 1 and
                    0 < w <= 1 and
                    0 < h <= 1
                ):
                    invalid.append(
                        (label.name, f"Line {line_number}: Invalid bounding box")
                    )

            except Exception as e:
                invalid.append(
                    (label.name, f"Line {line_number}: {e}")
                )

    return invalid, class_counter

# MAIN FUNCTION
def main():

    print("=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)

    train_images = count_files(TRAIN_IMAGES, [".jpg", ".png", ".jpeg"])
    val_images = count_files(VAL_IMAGES, [".jpg", ".png", ".jpeg"])

    print(f"Train Images      : {train_images}")
    print(f"Validation Images : {val_images}")

    missing_train = check_missing_labels(TRAIN_IMAGES, TRAIN_LABELS)
    missing_val = check_missing_labels(VAL_IMAGES, VAL_LABELS)

    print(f"\nMissing Train Labels : {len(missing_train)}")
    print(f"Missing Val Labels   : {len(missing_val)}")

    corrupted = (
        check_corrupted_images(TRAIN_IMAGES)
        + check_corrupted_images(VAL_IMAGES)
    )

    print(f"Corrupted Images     : {len(corrupted)}")

    invalid_train, train_counter = validate_labels(TRAIN_LABELS)
    invalid_val, val_counter = validate_labels(VAL_LABELS)

    invalid = invalid_train + invalid_val

    print(f"Invalid Labels       : {len(invalid)}")

    if invalid:
        print("\nInvalid Label Details:")
    for filename, reason in invalid:
        print(f" - {filename}: {reason}")

    print("\nClass Distribution")

    total_counter = train_counter + val_counter

    for class_id, count in total_counter.items():

        print(f"{CLASS_NAMES[class_id]:15} : {count}")

    print("\nValidation Finished Successfully")


if __name__ == "__main__":
    main()