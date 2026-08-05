from ultralytics import YOLO

# Load YOLO26
model = YOLO("runs/YOLO26_Baseline/weights/best.pt")

model.predict(
    source="dataset/processed/val/images/gg (136).jpg",
    conf=0.25,
    imgsz=640,
    save=True,
    project="comparison",
    name="v26_gg136"
)