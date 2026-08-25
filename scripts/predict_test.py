from ultralytics import YOLO
from pathlib import Path
import csv

# ============================================================
# CROP AND WEED DETECTION
# TEST SET PREDICTION
# ============================================================

MODEL_PATH = "models/crop_weed_yolov8/weights/best.pt"

TEST_IMAGES = "datasets/processed/images/test"

OUTPUT_DIR = Path("outputs/predictions")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CSV_FILE = OUTPUT_DIR / "test_predictions.csv"

CLASS_NAMES = {
    0: "crop",
    1: "weed"
}

print("=" * 60)
print("CROP AND WEED DETECTION - TEST PREDICTION")
print("=" * 60)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("\nLoading trained model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")

# ------------------------------------------------------------
# Run prediction
# ------------------------------------------------------------

results = model.predict(
    source=TEST_IMAGES,
    imgsz=416,
    conf=0.25,
    device="cpu",
    save=True,
    project="outputs/predictions",
    name="test_results",
    exist_ok=True,
    verbose=True
)

# ------------------------------------------------------------
# Create CSV
# ------------------------------------------------------------

with open(
    CSV_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "image",
        "crop_count",
        "weed_count",
        "total_detections"
    ])

    total_crops = 0
    total_weeds = 0

    for result in results:

        image_name = Path(
            result.path
        ).name

        crop_count = 0
        weed_count = 0

        if result.boxes is not None:

            for cls in result.boxes.cls:

                class_id = int(cls.item())

                if class_id == 0:
                    crop_count += 1

                elif class_id == 1:
                    weed_count += 1

        total_crops += crop_count
        total_weeds += weed_count

        writer.writerow([
            image_name,
            crop_count,
            weed_count,
            crop_count + weed_count
        ])

# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TEST PREDICTION SUMMARY")
print("=" * 60)

print(f"\nTest images        : {len(results)}")
print(f"Total crops        : {total_crops}")
print(f"Total weeds        : {total_weeds}")
print(
    f"Total detections   : "
    f"{total_crops + total_weeds}"
)

print("\nPrediction images saved to:")

print(
    Path(
        "outputs/predictions/test_results"
    ).resolve()
)

print("\nCSV report saved to:")

print(CSV_FILE.resolve())

print("\nPrediction completed successfully.")

print("=" * 60)