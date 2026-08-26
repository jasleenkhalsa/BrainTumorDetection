from pathlib import Path
import csv

import cv2
from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VAL_IMAGES = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "val"
    / "images"
)

VAL_LABELS = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "val"
    / "labels"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "YOLO26_Baseline"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "comparison"
    / "confidence_examples"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "selected_examples.csv"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 640

PREDICTION_CONFIDENCE = 0.20

MATCH_IOU_THRESHOLD = 0.50

TARGET_CONFIDENCES = [
    0.85,
    0.90,
    0.95,
]


# ============================================================
# CLASSES
# ============================================================

CLASS_NAMES = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary",
}

TUMOR_CLASSES = {
    0: "Glioma",
    1: "Meningioma",
    3: "Pituitary",
}


# ============================================================
# READ YOLO LABEL FILE
# ============================================================

def read_labels(label_path):

    annotations = []

    if not label_path.exists():
        return annotations

    with open(label_path, "r", encoding="utf-8") as file:

        for line in file:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(float(parts[0]))

            annotations.append(
                {
                    "class_id": class_id,
                    "x_center": float(parts[1]),
                    "y_center": float(parts[2]),
                    "width": float(parts[3]),
                    "height": float(parts[4]),
                }
            )

    return annotations


# ============================================================
# GT BOX TO PIXEL COORDINATES
# ============================================================

def ground_truth_xyxy(
    annotation,
    image_width,
    image_height,
):

    x_center = (
        annotation["x_center"]
        * image_width
    )

    y_center = (
        annotation["y_center"]
        * image_height
    )

    box_width = (
        annotation["width"]
        * image_width
    )

    box_height = (
        annotation["height"]
        * image_height
    )

    x1 = (
        x_center
        - box_width / 2
    )

    y1 = (
        y_center
        - box_height / 2
    )

    x2 = (
        x_center
        + box_width / 2
    )

    y2 = (
        y_center
        + box_height / 2
    )

    return [
        x1,
        y1,
        x2,
        y2,
    ]


# ============================================================
# IOU
# ============================================================

def calculate_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection_width = max(
        0,
        ix2 - ix1,
    )

    intersection_height = max(
        0,
        iy2 - iy1,
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    area_a = (
        max(0, ax2 - ax1)
        * max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1)
        * max(0, by2 - by1)
    )

    union = (
        area_a
        + area_b
        - intersection_area
    )

    if union <= 0:
        return 0.0

    return (
        intersection_area
        / union
    )


# ============================================================
# GET VALID DETECTIONS
# ============================================================

def get_valid_predictions(
    result,
    annotations,
    image_width,
    image_height,
):

    valid_predictions = []

    if result.boxes is None:
        return valid_predictions

    for prediction in result.boxes:

        predicted_class = int(
            prediction.cls.item()
        )

        if (
            predicted_class
            not in TUMOR_CLASSES
        ):
            continue

        confidence = float(
            prediction.conf.item()
        )

        predicted_box = [
            float(value)
            for value
            in prediction.xyxy[0].tolist()
        ]

        best_iou = 0.0

        for annotation in annotations:

            if (
                annotation["class_id"]
                != predicted_class
            ):
                continue

            gt_box = ground_truth_xyxy(
                annotation,
                image_width,
                image_height,
            )

            iou = calculate_iou(
                predicted_box,
                gt_box,
            )

            if iou > best_iou:
                best_iou = iou

        if (
            best_iou
            >= MATCH_IOU_THRESHOLD
        ):

            valid_predictions.append(
                {
                    "class_id":
                        predicted_class,

                    "class_name":
                        TUMOR_CLASSES[
                            predicted_class
                        ],

                    "confidence":
                        confidence,

                    "bbox":
                        predicted_box,

                    "iou":
                        best_iou,
                }
            )

    return valid_predictions


# ============================================================
# FIND CLOSEST EXAMPLES
# ============================================================

def find_examples(model):

    candidates = {
        class_id: []
        for class_id
        in TUMOR_CLASSES
    }

    image_paths = sorted(
        list(
            VAL_IMAGES.glob("*.jpg")
        )
        + list(
            VAL_IMAGES.glob("*.jpeg")
        )
        + list(
            VAL_IMAGES.glob("*.png")
        )
    )

    print(
        f"Validation images found: "
        f"{len(image_paths)}"
    )

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        height, width = (
            image.shape[:2]
        )

        label_path = (
            VAL_LABELS
            / f"{image_path.stem}.txt"
        )

        annotations = read_labels(
            label_path
        )

        tumor_annotations = [
            annotation
            for annotation
            in annotations
            if annotation["class_id"]
            in TUMOR_CLASSES
        ]

        if not tumor_annotations:
            continue

        result = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=PREDICTION_CONFIDENCE,
            device=0,
            verbose=False,
        )[0]

        valid_predictions = (
            get_valid_predictions(
                result,
                tumor_annotations,
                width,
                height,
            )
        )

        for prediction in valid_predictions:

            prediction[
                "image_path"
            ] = image_path

            prediction[
                "label_path"
            ] = label_path

            candidates[
                prediction["class_id"]
            ].append(
                prediction
            )

        if index % 100 == 0:

            print(
                f"Processed "
                f"{index}/"
                f"{len(image_paths)}"
            )

    selected = {}

    for class_id in TUMOR_CLASSES:

        selected[class_id] = {}

        used_images = set()

        for target in TARGET_CONFIDENCES:

            available = [
                item
                for item
                in candidates[class_id]
                if item[
                    "image_path"
                ].name
                not in used_images
            ]

            if not available:
                available = (
                    candidates[class_id]
                )

            if not available:

                selected[
                    class_id
                ][target] = None

                continue

            best = min(
                available,
                key=lambda item:
                    abs(
                        item["confidence"]
                        - target
                    ),
            )

            selected[
                class_id
            ][target] = best

            used_images.add(
                best[
                    "image_path"
                ].name
            )

    return selected


# ============================================================
# SAVE INDIVIDUAL IMAGE
# ============================================================

def save_detection_image(
    example,
    target,
):

    class_name = (
        example["class_name"]
    )

    confidence = (
        example["confidence"]
    )

    iou = (
        example["iou"]
    )

    image_path = (
        example["image_path"]
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        return None

    x1, y1, x2, y2 = [
        int(round(value))
        for value
        in example["bbox"]
    ]

    # Draw detection box
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3,
    )

    label = (
        f"{class_name} "
        f"{confidence:.3f}"
    )

    cv2.putText(
        image,
        label,
        (
            max(10, x1),
            max(30, y1 - 10),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # ADD INFORMATION AT TOP
    # --------------------------------------------------------

    header_height = 115

    image = cv2.copyMakeBorder(
        image,
        header_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    cv2.putText(
        image,
        class_name,
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        (
            f"Target Confidence: "
            f"{target * 100:.0f}%"
        ),
        (15, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        (
            f"Actual Confidence: "
            f"{confidence * 100:.2f}%"
        ),
        (15, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        (
            f"IoU: {iou:.3f}"
            f" | Raw Image: "
            f"{image_path.name}"
        ),
        (15, 101),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # SAVE TO CLASS FOLDER
    # --------------------------------------------------------

    class_folder = (
        OUTPUT_DIR
        / class_name
    )

    class_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_name = (
        f"{class_name}"
        f"_target{int(target * 100)}"
        f"_actual{confidence * 100:.2f}"
        f"_{image_path.stem}.jpg"
    )

    output_path = (
        class_folder
        / output_name
    )

    cv2.imwrite(
        str(output_path),
        image,
    )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "YOLO26 Individual Confidence Examples"
    )
    print("=" * 75)

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{MODEL_PATH}"
        )

    if not VAL_IMAGES.exists():

        raise FileNotFoundError(
            f"Validation images not found:\n"
            f"{VAL_IMAGES}"
        )

    if not VAL_LABELS.exists():

        raise FileNotFoundError(
            f"Validation labels not found:\n"
            f"{VAL_LABELS}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\nLoading model:\n"
        f"{MODEL_PATH}\n"
    )

    model = YOLO(
        str(MODEL_PATH)
    )

    selected = find_examples(
        model
    )

    csv_rows = []

    print("\n")
    print("=" * 75)
    print("SELECTED IMAGES")
    print("=" * 75)

    for class_id, class_name in (
        TUMOR_CLASSES.items()
    ):

        print(
            f"\n{class_name}"
        )

        for target in (
            TARGET_CONFIDENCES
        ):

            example = (
                selected[
                    class_id
                ][target]
            )

            if example is None:

                print(
                    f"Target "
                    f"{target * 100:.0f}%"
                    f" -> No valid example"
                )

                continue

            output_path = (
                save_detection_image(
                    example,
                    target,
                )
            )

            print(
                f"\nTarget: "
                f"{target * 100:.0f}%"
            )

            print(
                f"Raw image: "
                f"{example['image_path'].name}"
            )

            print(
                f"Raw path : "
                f"{example['image_path']}"
            )

            print(
                f"Label file: "
                f"{example['label_path'].name}"
            )

            print(
                f"Actual confidence: "
                f"{example['confidence'] * 100:.2f}%"
            )

            print(
                f"IoU: "
                f"{example['iou']:.4f}"
            )

            print(
                f"Detection image saved: "
                f"{output_path}"
            )

            csv_rows.append(
                [
                    class_name,

                    f"{target * 100:.0f}%",

                    (
                        f"{example['confidence'] * 100:.2f}%"
                    ),

                    f"{example['iou']:.4f}",

                    example[
                        "image_path"
                    ].name,

                    str(
                        example[
                            "image_path"
                        ]
                    ),

                    example[
                        "label_path"
                    ].name,

                    str(
                        example[
                            "label_path"
                        ]
                    ),

                    str(
                        output_path
                    ),
                ]
            )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "Tumor Class",
                "Target Confidence",
                "Actual Confidence",
                "IoU",
                "Raw Image Filename",
                "Raw Image Full Path",
                "Label Filename",
                "Label Full Path",
                "Generated Detection Image",
            ]
        )

        writer.writerows(
            csv_rows
        )

    print("\n")
    print("=" * 75)
    print("DONE")
    print("=" * 75)

    print(
        f"\nIndividual images saved in:\n"
        f"{OUTPUT_DIR}"
    )

    print(
        f"\nMatching CSV saved as:\n"
        f"{OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()