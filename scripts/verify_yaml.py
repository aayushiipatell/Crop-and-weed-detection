from pathlib import Path
import yaml

YAML_FILE = Path("datasets/processed/data.yaml")

print("=" * 60)
print("YOLO DATASET CONFIGURATION VERIFICATION")
print("=" * 60)

if not YAML_FILE.exists():
    print("\nERROR: data.yaml not found")
    exit()

with open(YAML_FILE, "r") as file:
    data = yaml.safe_load(file)

print("\nDataset path:")
print(data["path"])

print("\nTraining path:")
print(data["train"])

print("\nValidation path:")
print(data["val"])

print("\nTesting path:")
print(data["test"])

print("\nNumber of classes:")
print(data["nc"])

print("\nClass names:")
print(data["names"])

# ------------------------------------------------------------
# Verify directories
# ------------------------------------------------------------

base_path = Path(data["path"])

train_path = base_path / data["train"]
val_path = base_path / data["val"]
test_path = base_path / data["test"]

print("\n" + "-" * 60)
print("DIRECTORY CHECK")
print("-" * 60)

print("Train directory exists :", train_path.exists())
print("Val directory exists   :", val_path.exists())
print("Test directory exists  :", test_path.exists())

# ------------------------------------------------------------
# Count images
# ------------------------------------------------------------

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

train_images = [
    p for p in train_path.iterdir()
    if p.suffix.lower() in extensions
]

val_images = [
    p for p in val_path.iterdir()
    if p.suffix.lower() in extensions
]

test_images = [
    p for p in test_path.iterdir()
    if p.suffix.lower() in extensions
]

print("\n" + "-" * 60)
print("IMAGE COUNTS")
print("-" * 60)

print("Train images :", len(train_images))
print("Val images   :", len(val_images))
print("Test images  :", len(test_images))

# ------------------------------------------------------------
# Final result
# ------------------------------------------------------------

if (
    train_path.exists()
    and val_path.exists()
    and test_path.exists()
    and data["nc"] == 2
    and data["names"][0] == "crop"
    and data["names"][1] == "weed"
):

    print("\n✅ data.yaml verification successful.")

else:

    print("\n❌ data.yaml contains configuration issues.")

print("=" * 60)