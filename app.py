import streamlit as st
import numpy as np
import gdown
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🍃",
    layout="wide"
)

# Custom CSS
st.markdown(
    """
    <style>
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #e0f7fa 0%, #e8f5e9 100%);
    }

    /* Override text color for all text elements with more specificity */
    html, body, [class^="css"] {
        color: #1b5e20 !important;
        font-weight: 600;
    }

    /* Title styling */
    .title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1b5e20;
        margin-top: 1rem;
    }

    /* Subheaders */
    .stMarkdown h2, .stMarkdown h3 {
        color: #256029 !important;
        font-weight: bold;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #388e3c;
        color: white !important;
        font-size: 1.2rem;
        padding: 0.6rem 1.2rem;
        border-radius: 12px;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #2e7d32;
    }

    /* Sidebar text and headings */
    .sidebar .sidebar-content, .css-1d391kg, .css-1v3fvcr {
        color: #1b5e20 !important;
    }

    /* Footer */
    footer, footer * {
        color: #1b5e20 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Constants
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

# Model loader with caching
@st.cache_resource
def load_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading trained model..."):
            gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=True)
    return load_model(MODEL_PATH)

# Load the model
try:
    model = load_model_from_drive()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")

# Header
st.markdown("<div class='title'>🌿 Plant Disease Detection</div>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("About")
    st.write(
        "Upload a clear photo of a leaf, and our AI-powered model will identify the disease for you in seconds!"
    )
    st.markdown("---")
    st.subheader("Possible Classes:")
    for cls in CLASS_NAMES:
        st.write(f"• {cls.replace('_', ' ')}")
    st.markdown("---")
    st.write("Developed by Your Name 🌱")

# Main layout
upload_col, result_col = st.columns(2)

with upload_col:
    st.subheader("1. Upload Leaf Image")
    uploaded_file = st.file_uploader(
        label="",
        type=["jpg", "jpeg", "png"],
        help="Choose a leaf image file"
    )

with result_col:
    st.subheader("2. Prediction")
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Leaf", use_column_width=True)

    if uploaded_file and st.button("Detect Disease 🍃"):
        try:
            with st.spinner("🔬 Analyzing image..."):
                img_resized = img.resize((224, 224))
                img_array = image.img_to_array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                preds = model.predict(img_array)
                pred_idx = int(np.argmax(preds, axis=1)[0])
                pred_label = CLASS_NAMES[pred_idx].replace("_", " ")
                confidence = float(np.max(preds)) * 100
            st.success(f"**{pred_label}** detected with {confidence:.2f}% confidence! 🌿")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    elif not uploaded_file:
        st.info("Please upload an image to begin.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #555;'>&copy; 2025 Plant Disease Detection | Powered by TensorFlow & Streamlit</p>",
    unsafe_allow_html=True
)
