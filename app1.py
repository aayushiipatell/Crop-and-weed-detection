import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
from pathlib import Path

# ============================================================
# CROP & WEED DETECTION SYSTEM
# Streamlit Application
# ============================================================

st.set_page_config(
    page_title="Crop & Weed Detection",
    page_icon="🌱",
    layout="wide"
)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

MODEL_PATH = Path("models/best.pt")

# ------------------------------------------------------------
# Load Model
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.title("🌱 Crop & Weed Detection System")

st.markdown(
    """
    Upload an agricultural image to automatically detect **crops and weeds**
    using a trained YOLOv8 object detection model.
    """
)

st.divider()

# ------------------------------------------------------------
# Check Model
# ------------------------------------------------------------

if not MODEL_PATH.exists():

    st.error(
        "Model file not found. Please make sure "
        "`models/best.pt` exists."
    )

    st.stop()

model = load_model()

# ------------------------------------------------------------
# Upload Image
# ------------------------------------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload an agricultural image",
    type=["jpg", "jpeg", "png"]
)

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("📷 Uploaded Image")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Original Image",
            use_container_width=True
        )

    # --------------------------------------------------------
    # Save temporary image
    # --------------------------------------------------------

    with tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    ) as temp_file:

        image.save(temp_file.name)
        temp_path = temp_file.name

    # --------------------------------------------------------
    # Run YOLO prediction
    # --------------------------------------------------------

    with st.spinner("🔍 Detecting crops and weeds..."):

        results = model.predict(
            source=temp_path,
            imgsz=416,
            conf=0.25,
            device="cpu",
            verbose=False
        )

    result = results[0]

    # --------------------------------------------------------
    # Annotated Image
    # --------------------------------------------------------

    annotated_image = result.plot()

    with col2:

        st.subheader("🎯 Detection Result")

        st.image(
            annotated_image,
            caption="Detected Crops and Weeds",
            use_container_width=True
        )

    # --------------------------------------------------------
    # Count Objects
    # --------------------------------------------------------

    crop_count = 0
    weed_count = 0

    if result.boxes is not None:

        for cls in result.boxes.cls:

            class_id = int(cls.item())

            if class_id == 0:
                crop_count += 1

            elif class_id == 1:
                weed_count += 1

    total_count = crop_count + weed_count

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    st.divider()

    st.subheader("📊 Detection Summary")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "🌱 Crops Detected",
            crop_count
        )

    with metric2:
        st.metric(
            "🌿 Weeds Detected",
            weed_count
        )

    with metric3:
        st.metric(
            "📦 Total Objects",
            total_count
        )

    # --------------------------------------------------------
    # Detection Details
    # --------------------------------------------------------

    if result.boxes is not None and len(result.boxes) > 0:

        st.subheader("🔎 Detection Details")

        for i, box in enumerate(result.boxes):

            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            class_name = (
                "Crop"
                if class_id == 0
                else "Weed"
            )

            st.write(
                f"**Detection {i + 1}:** "
                f"{class_name} — "
                f"Confidence: {confidence:.2%}"
            )

    else:

        st.warning(
            "No crops or weeds were detected in this image."
        )

else:

    st.info(
        "👆 Upload an agricultural image to begin detection."
    )

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------

st.divider()

st.caption(
    "Crop & Weed Detection System | YOLOv8 | "
    "Object Detection"
)