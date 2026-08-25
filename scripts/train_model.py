from ultralytics import YOLO
from pathlib import Path

DATA_YAML = "datasets/processed/data.yaml"

OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("CROP AND WEED DETECTION - FINAL TRAINING")
print("=" * 60)

model = YOLO("yolov8n.pt")

results = model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=416,
    batch=8,
    device="cpu",
    workers=0,
    project=str(OUTPUT_DIR.resolve()),
    name="crop_weed_yolov8",
    exist_ok=True,
    pretrained=True,
    patience=10,
    cache=False,
    verbose=True
)

print("\n" + "=" * 60)
print("FINAL TRAINING COMPLETED")
print("=" * 60)

best_model = (
    OUTPUT_DIR /
    "crop_weed_yolov8" /
    "weights" /
    "best.pt"
)

last_model = (
    OUTPUT_DIR /
    "crop_weed_yolov8" /
    "weights" /
    "last.pt"
)

print("\nBest model:")
print(best_model.resolve())

print("\nLast model:")
print(last_model.resolve())

print("=" * 60)