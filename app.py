import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# -----------------------------
# UI
# -----------------------------
st.title("🪖 Helmet Detection App")
st.write("Upload an image to detect helmets.")

# Confidence slider
conf = st.slider("Confidence Threshold", 0.1, 1.0, 0.3)

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        # Predict
        results = model.predict(
            source=img_array,
            conf=conf,
            imgsz=640,
            verbose=False
        )

        annotated_img = results[0].plot()

        # -----------------------------
        # 🔥 2-COLUMN LAYOUT
        # -----------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original Image", width="stretch")

        with col2:
            st.image(annotated_img, caption="Detection Result", width="stretch")

        # -----------------------------
        # Detection Summary
        # -----------------------------
        # boxes = results[0].boxes

        # if boxes is not None:
        #     classes = boxes.cls.cpu().numpy()
        #     names = model.names

        #     helmet_count = sum(1 for c in classes if names[int(c)] == "helmet")
        #     person_count = sum(1 for c in classes if names[int(c)] == "person")

        #     st.write("### 📊 Detection Summary")
        #     st.write(f"🪖 Helmets: {helmet_count}")
        #     st.write(f"👤 Persons: {person_count}")

    except Exception as e:
        st.error(f"Error processing image: {e}")