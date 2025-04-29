import streamlit as st
import numpy as np
import gdown
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ✅ Page Config
st.set_page_config(
    page_title="PhytoScan AI - Plant Disease Detection",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ✅ Model Info
MODEL_PATH = "plant_disease_modelfinal2.h5"
FILE_ID = "1tDt1NSWyfkqtFzh91KJQtPNVl5mbc2QG"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# ✅ Class Names
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# ✅ Download model if missing
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading AI model..."):
        gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=True)

# ✅ Cached Model Loader
@st.cache_resource(show_spinner=False)
def load_cached_model():
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        raise

# ✅ Display Prediction Results
def display_results(predicted_class, confidence):
    st.markdown("### 🧾 Diagnosis Result")
    st.success(f"**Predicted Disease:** {predicted_class.replace('_', ' ')}")
    st.info(f"**Confidence:** {confidence:.2f}%")

# ✅ Main App Logic
def main():
    # App Header
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2917/2917996.png", width=80)
    with col2:
        st.title("PhytoScan AI")
        st.markdown("**AI-Powered Plant Health Diagnosis System**")

    st.markdown("---")

    # File Upload
    st.subheader("🌿 Upload Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        # Image Preview
        with st.expander("📸 Image Preview", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                img = Image.open(uploaded_file)
                st.image(img, caption="Original Image", use_column_width=True)
            with col2:
                processed_img = img.resize((224, 224))
                st.image(processed_img, caption="Processed (224x224)", use_column_width=True)

        # Prediction Trigger
        if st.button("🔍 Analyze Now", use_container_width=True):
            try:
                model = load_cached_model()

                with st.status("🧠 AI Analysis in Progress...", expanded=True) as status:
                    st.write("Preprocessing image...")
                    img_array = image.img_to_array(processed_img)
                    img_array = np.expand_dims(img_array, axis=0) / 255.0

                    st.write("Running prediction...")
                    predictions = model.predict(img_array)
                    predicted_class = CLASS_NAMES[np.argmax(predictions)]
                    confidence = np.max(predictions) * 100

                    status.update(label="✅ Analysis Complete", state="complete")

                display_results(predicted_class, confidence)

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

# ✅ Run App
if __name__ == "__main__":
    main()
