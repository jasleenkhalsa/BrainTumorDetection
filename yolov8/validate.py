from ultralytics import YOLO


def main():
    model = YOLO("runs/YOLOv8_Baseline/weights/best.pt")

    metrics = model.val(
        data="dataset/data.yaml",
        imgsz=640,
        batch=4,
        workers=0,
        device=0,
        project="runs",
        name="YOLOv8_Validation"
    )

    print("\nValidation completed successfully.")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()