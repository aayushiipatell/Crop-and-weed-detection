from pathlib import Path
from collections import Counter
from PIL import Image

# ============================================================
# CROP AND WEED DETECTION
# DATASET VERIFICATION
# ============================================================

# Actual dataset location
DATASET_DIR = Path("datasets/agri_data/data")

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

print("=" * 60)
print("CROP AND WEED DETECTION - DATASET VERIFICATION")
print("=" * 60)

# ------------------------------------------------------------
# Check dataset directory
# ------------------------------------------------------------

if not DATASET_DIR.exists():
    print(f"\nERROR: Dataset directory not found:")
    print(DATASET_DIR)
    exit()

print(f"\nDataset location:")
print(DATASET_DIR.resolve())

# ------------------------------------------------------------
# Find images and labels
# ------------------------------------------------------------

images = [
    p for p in DATASET_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]

labels = [
    p for p in DATASET_DIR.iterdir()
    if p.is_file() and p.suffix.lower() == ".txt"
]

print(f"\nTotal images found : {len(images)}")
print(f"Total labels found : {len(labels)}")

# ------------------------------------------------------------
# Match images and labels
# ------------------------------------------------------------

image_names = {p.stem for p in images}
label_names = {p.stem for p in labels}

images_without_labels = image_names - label_names
labels_without_images = label_names - image_names

print("\n" + "-" * 60)
print("IMAGE-LABEL MATCHING")
print("-" * 60)

print(f"Images without labels : {len(images_without_labels)}")
print(f"Labels without images : {len(labels_without_images)}")

if images_without_labels:
    print("\nImages without labels:")
    for name in sorted(images_without_labels):
        print("  ", name)

if labels_without_images:
    print("\nLabels without images:")
    for name in sorted(labels_without_images):
        print("  ", name)

# ------------------------------------------------------------
# Validate YOLO annotations
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("YOLO ANNOTATION VALIDATION")
print("-" * 60)

invalid_lines = []
invalid_files = []

class_counter = Counter()
total_annotations = 0

for label_file in labels:

    try:
        with open(label_file, "r") as f:
            lines = f.readlines()

        for line_number, line in enumerate(lines, start=1):

            line = line.strip()

            # Empty label file
            if not line:
                continue

            parts = line.split()

            # YOLO annotation must have 5 values
            if len(parts) != 5:
                invalid_lines.append(
                    (label_file.name, line_number, "Expected 5 values")
                )
                continue

            try:
                class_id = int(parts[0])

                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

            except ValueError:
                invalid_lines.append(
                    (label_file.name, line_number, "Non-numeric value")
                )
                continue

            # Validate class ID
            if class_id < 0:
                invalid_lines.append(
                    (label_file.name, line_number, "Invalid class ID")
                )

            # Validate bounding box
            if not (0 <= x_center <= 1):
                invalid_lines.append(
                    (label_file.name, line_number, "Invalid x_center")
                )

            if not (0 <= y_center <= 1):
                invalid_lines.append(
                    (label_file.name, line_number, "Invalid y_center")
                )

            if not (0 < width <= 1):
                invalid_lines.append(
                    (label_file.name, line_number, "Invalid width")
                )

            if not (0 < height <= 1):
                invalid_lines.append(
                    (label_file.name, line_number, "Invalid height")
                )

            class_counter[class_id] += 1
            total_annotations += 1

    except Exception as e:

        invalid_files.append(
            (label_file.name, str(e))
        )

print(f"Total annotations        : {total_annotations}")
print(f"Invalid annotation files : {len(invalid_files)}")
print(f"Invalid annotation lines  : {len(invalid_lines)}")

# ------------------------------------------------------------
# Class distribution
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("CLASS DISTRIBUTION")
print("-" * 60)

if class_counter:

    for class_id, count in sorted(class_counter.items()):
        print(f"Class {class_id}: {count} annotations")

else:

    print("No annotations found.")

# ------------------------------------------------------------
# Read classes.txt
# ------------------------------------------------------------

CLASSES_FILE = Path("datasets/classes.txt")

print("\n" + "-" * 60)
print("CLASS NAMES")
print("-" * 60)

class_names = []

if CLASSES_FILE.exists():

    with open(CLASSES_FILE, "r") as f:

        for line in f:

            line = line.strip()

            if line:
                class_names.append(line)

    for index, name in enumerate(class_names):
        print(f"Class {index}: {name}")

else:

    print("classes.txt not found.")

# ------------------------------------------------------------
# Image dimensions
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("IMAGE DIMENSIONS")
print("-" * 60)

dimension_counter = Counter()
corrupted_images = []

for image_path in images:

    try:

        with Image.open(image_path) as img:

            dimension_counter[img.size] += 1

    except Exception as e:

        corrupted_images.append(
            (image_path.name, str(e))
        )

for dimension, count in dimension_counter.items():

    print(f"{dimension}: {count} images")

print(
    f"\nCorrupted/unreadable images: "
    f"{len(corrupted_images)}"
)

# ------------------------------------------------------------
# Dataset Summary
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATASET VERIFICATION SUMMARY")
print("=" * 60)

print(f"Images                   : {len(images)}")
print(f"Labels                   : {len(labels)}")
print(f"Images without labels    : {len(images_without_labels)}")
print(f"Labels without images    : {len(labels_without_images)}")
print(f"Total annotations        : {total_annotations}")
print(f"Invalid annotation lines : {len(invalid_lines)}")
print(f"Corrupted images         : {len(corrupted_images)}")

print("\nClass IDs detected:")

for class_id, count in sorted(class_counter.items()):

    name = (
        class_names[class_id]
        if class_id < len(class_names)
        else "Unknown"
    )

    print(
        f"  Class {class_id} ({name}): "
        f"{count} annotations"
    )

# ------------------------------------------------------------
# Final verification status
# ------------------------------------------------------------

if (
    len(images_without_labels) == 0
    and len(labels_without_images) == 0
    and len(invalid_lines) == 0
    and len(invalid_files) == 0
    and len(corrupted_images) == 0
):

    print("\n✅ DATASET VERIFICATION PASSED")

else:

    print("\n⚠️ DATASET CONTAINS ISSUES")

print("=" * 60)