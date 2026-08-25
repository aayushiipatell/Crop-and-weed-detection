from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt

DATASET = Path("datasets/processed")

IMAGE_DIR = DATASET / "images" / "train"
LABEL_DIR = DATASET / "labels" / "train"

CLASS_NAMES = {
    0: "crop",
    1: "weed"
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

images = [
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in IMAGE_EXTENSIONS
]

if not images:
    print("No training images found.")
    exit()

# Select 5 random images
sample_images = random.sample(
    images,
    min(5, len(images))
)

for image_path in sample_images:

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width = image.shape[:2]

    if label_path.exists():

        with open(label_path, "r") as file:

            for line in file:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                class_id = int(parts[0])

                x_center = float(parts[1])
                y_center = float(parts[2])
                box_width = float(parts[3])
                box_height = float(parts[4])

                # Convert normalized coordinates
                x_center *= width
                y_center *= height
                box_width *= width
                box_height *= height

                x1 = int(x_center - box_width / 2)
                y1 = int(y_center - box_height / 2)

                x2 = int(x_center + box_width / 2)
                y2 = int(y_center + box_height / 2)

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

                label = CLASS_NAMES.get(
                    class_id,
                    f"class_{class_id}"
                )

                cv2.putText(
                    image,
                    label,
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(image_path.name)
    plt.axis("off")
    plt.show()