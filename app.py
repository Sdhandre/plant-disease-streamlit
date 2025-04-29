import streamlit as st
import numpy as np
import gdown
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ✅ Must be FIRST Streamlit command
st.set_page_config(page_title="Plant Disease Detection", layout="centered")

# ----------- Constants -------------
MODEL_PATH = "plant_disease_modelfinal2.h5"
FILE_ID = "1tDt1NSWyfkqtFzh91KJQtPNVl5mbc2QG"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"
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

# ----------- Model Loader with Cache -------------
@st.cache_resource
def load_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model..."):
            gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

# Load the model and confirm
try:
    model = load_model_from_drive()
    st.write("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Model failed to load: {e}")

# ----------- UI Setup -------------
st.title("🌿 Plant Disease Detection App")
st.markdown("Upload a leaf image to detect the disease.")
st.write("✅ UI loaded")

# ----------- Upload Image -------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Leaf Image', use_column_width=True)
    st.write("✅ File uploaded")

    if st.button("Predict"):
        st.write("🔍 Predict button clicked")
        try:
            with st.spinner("Analyzing..."):
                img_resized = img.resize((224, 224))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0  # Normalize if required by model

                prediction = model.predict(img_array)
                predicted_class = CLASS_NAMES[np.argmax(prediction)]

                st.success(f"🧠 Predicted: **{predicted_class}**")
        except Exception as e:
            st.error(f"⚠️ Prediction error: {e}")
