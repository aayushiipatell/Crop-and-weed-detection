from pathlib import Path
import random
import shutil

# ============================================================
# CROP AND WEED DETECTION
# DATASET SPLITTING
# ============================================================

SOURCE_DIR = Path("datasets/agri_data/data")

OUTPUT_DIR = Path("datasets/processed")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

# Dataset split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

# Reproducibility
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# ------------------------------------------------------------
# Check source dataset
# ------------------------------------------------------------

if not SOURCE_DIR.exists():

    print("ERROR: Source dataset not found:")
    print(SOURCE_DIR)
    exit()

# ------------------------------------------------------------
# Create output directories
# ------------------------------------------------------------

for split in ["train", "val", "test"]:

    (OUTPUT_DIR / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_DIR / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )

# ------------------------------------------------------------
# Find images
# ------------------------------------------------------------

images = [
    file
    for file in SOURCE_DIR.iterdir()
    if file.is_file()
    and file.suffix.lower() in IMAGE_EXTENSIONS
]

print("=" * 60)
print("DATASET SPLITTING")
print("=" * 60)

print(f"\nTotal images found: {len(images)}")

# ------------------------------------------------------------
# Verify corresponding labels
# ------------------------------------------------------------

valid_pairs = []

missing_labels = []

for image in images:

    label = SOURCE_DIR / f"{image.stem}.txt"

    if label.exists():

        valid_pairs.append((image, label))

    else:

        missing_labels.append(image.name)

if missing_labels:

    print("\nWARNING: Images without labels:")

    for image in missing_labels:
        print(" ", image)

print(f"\nValid image-label pairs: {len(valid_pairs)}")

# ------------------------------------------------------------
# Shuffle dataset
# ------------------------------------------------------------

random.shuffle(valid_pairs)

# ------------------------------------------------------------
# Calculate split sizes
# ------------------------------------------------------------

total = len(valid_pairs)

train_end = int(total * TRAIN_RATIO)

val_end = train_end + int(total * VAL_RATIO)

train_data = valid_pairs[:train_end]

val_data = valid_pairs[train_end:val_end]

test_data = valid_pairs[val_end:]

# ------------------------------------------------------------
# Copy files
# ------------------------------------------------------------

def copy_split(data, split_name):

    image_output = OUTPUT_DIR / "images" / split_name

    label_output = OUTPUT_DIR / "labels" / split_name

    for image, label in data:

        shutil.copy2(
            image,
            image_output / image.name
        )

        shutil.copy2(
            label,
            label_output / label.name
        )

    print(
        f"{split_name.capitalize():10} : "
        f"{len(data)} images"
    )


print("\nDataset split:")

copy_split(train_data, "train")

copy_split(val_data, "val")

copy_split(test_data, "test")

# ------------------------------------------------------------
# Final verification
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("SPLIT SUMMARY")
print("=" * 60)

print(f"Training images   : {len(train_data)}")
print(f"Validation images : {len(val_data)}")
print(f"Testing images    : {len(test_data)}")
print(f"Total             : {len(train_data) + len(val_data) + len(test_data)}")

print("\nOutput location:")
print(OUTPUT_DIR.resolve())

print("\nDataset splitting completed successfully.")
print("=" * 60)