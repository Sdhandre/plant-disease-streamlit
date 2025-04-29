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
    page_icon="🌌",
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

# ✅ Cached Model Loader (Fixed Indentation)
@st.cache_resource(show_spinner=False)
def load_cached_model():
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        raise

# ✅ Display Prediction Results
def display_results(predicted_class, confidence):
    st.markdown(f"""
    <div class="result-card">
        <div class="holographic-overlay"></div>
        <div class="cyberpunk-text glow">
            <h3>🌌 PHYTOSCAN DIAGNOSIS</h3>
            <div class="pulse-divider"></div>
            <p class="diagnosis">🔮 {predicted_class.replace('_', ' ')}</p>
            <p class="confidence">⚡ {confidence:.2f}% CERTAINTY</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ✅ Main App Logic
def main():
    # -- Cosmic CSS Fixes --
    st.markdown("""
    <style>
        @keyframes nebulaPulse {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .block-container {
            background: linear-gradient(-45deg, #0a0a2e, #1a1a4a, #2d0a3d, #4a1a4a);
            animation: nebulaPulse 15s ease infinite;
            border: 2px solid #0ff;
            box-shadow: 0 0 40px #0ff4;
        }
        
        .result-card {
            position: relative;
            background: rgba(10, 10, 40, 0.9);
            border-radius: 15px;
            padding: 2rem;
            margin: 2rem 0;
            backdrop-filter: blur(10px);
        }
        
        .cyberpunk-text {
            font-family: 'Orbitron', sans-serif;
            color: #0ff;
            text-shadow: 0 0 10px #0ff;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header Section
    st.markdown("""
    <div class="cyberpunk-text" style="text-align:center; padding:2rem;">
        <h1 style="font-size:3em;">🪐 PHYTOSCAN AI</h1>
        <p style="font-size:1.2em;">N E U R A L • P L A N T • D I A G N O S T I C S</p>
    </div>
    """, unsafe_allow_html=True)

    # File Uploader
    uploaded_file = st.file_uploader(
        "🌀 UPLOAD LEAF SPECIMEN",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of plant leaves"
    )

    if uploaded_file:
        # Image Processing
        img = Image.open(uploaded_file).convert('RGB')
        processed_img = img.resize((256, 256))
        
        with st.expander("🔍 IMAGE ANALYSIS", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="ORIGINAL SPECIMEN", use_column_width=True)
            with col2:
                st.image(processed_img, caption="ENHANCED VIEW", use_column_width=True)

        # Prediction
        if st.button("🚀 INITIATE QUANTUM SCAN", use_container_width=True):
            with st.spinner("🔭 ANALYZING BIO-PATTERNS..."):
                try:
                    model = load_cached_model()
                    img_array = image.img_to_array(processed_img) / 255.0
                    prediction = model.predict(np.expand_dims(img_array, axis=0))
                    
                    predicted_class = CLASS_NAMES[np.argmax(prediction)]
                    confidence = np.max(prediction) * 100
                    
                    display_results(predicted_class, confidence)
                    
                except Exception as e:
                    st.error(f"⚠️ SCAN FAILURE: {str(e)}")

if __name__ == "__main__":
    main()
