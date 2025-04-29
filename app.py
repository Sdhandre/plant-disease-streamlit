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

# ✅ Cached Model Loader
@st.cache_resource(show_spinner=False)
def load_cached_model():
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        raise

# ✅ Display Prediction Results in Card Style
def display_results(predicted_class, confidence):
    st.markdown(
        f"""
        <div class="result-card">
            <div class="holographic-effect"></div>
            <div class="card-content">
                <h3 class="cyberpunk">DIAGNOSIS REPORT</h3>
                <div class="glowing-divider"></div>
                <p class="prediction-text">🌿 {predicted_class.replace('_', ' ')}</p>
                <p class="confidence-text">⚡ {confidence:.2f}% CERTAINTY</p>
            </div>
            <div class="particles">
                <div class="particle"></div>
                <div class="particle"></div>
                <div class="particle"></div>
            </div>
        </div>
        """, unsafe_allow_html=True
    )

# ✅ Main App Logic
def main():
    # Cosmic CSS
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&family=Ubuntu+Mono&display=swap');
            
            /* Cosmic Background */
            .block-container {
                background: linear-gradient(-45deg, #0a0a2e, #1a1a4a, #2d0a3d, #4a1a4a);
                background-size: 400% 400%;
                animation: gradientBG 15s ease infinite;
                border-radius: 20px;
                position: relative;
                overflow: hidden;
            }
            
            @keyframes gradientBG {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
            
            /* Holographic Particles */
            .block-container:before {
                content: '';
                position: absolute;
                width: 200%;
                height: 200%;
                background-image: radial-gradient(circle, #00ffaa10 20%, transparent 20%);
                background-size: 30px 30px;
                animation: particles 20s linear infinite;
                opacity: 0.3;
            }
            
            @keyframes particles {
                0% { transform: translateY(0) translateX(0); }
                100% { transform: translateY(-1000px) translateX(-500px); }
            }
            
            /* Cyberpunk Title */
            .cyberpunk {
                font-family: 'Orbitron', sans-serif;
                color: #0ff;
                text-shadow: 0 0 10px #0ff,
                             0 0 20px #0ff,
                             0 0 30px #0ff;
                text-align: center;
                letter-spacing: 2px;
            }
            
            /* Holographic Result Card */
            .result-card {
                background: rgba(10, 10, 40, 0.8);
                border: 2px solid #0ff;
                border-radius: 15px;
                padding: 25px;
                margin: 20px 0;
                position: relative;
                backdrop-filter: blur(10px);
                box-shadow: 0 0 30px #0ff4;
                transform-style: preserve-3d;
                transition: all 0.3s ease;
            }
            
            .result-card:hover {
                transform: perspective(1000px) rotateX(5deg) rotateY(5deg);
                box-shadow: 0 0 50px #0ff8;
            }
            
            .glowing-divider {
                height: 2px;
                background: linear-gradient(90deg, transparent, #0ff, transparent);
                margin: 15px 0;
                box-shadow: 0 0 10px #0ff;
            }
            
            /* Neon Button Style */
            .stButton button {
                background: linear-gradient(45deg, #0ff, #f0f);
                color: #000 !important;
                font-family: 'Orbitron', sans-serif;
                border: none;
                border-radius: 8px;
                padding: 15px 30px;
                cursor: pointer;
                position: relative;
                overflow: hidden;
                transition: all 0.3s ease;
                box-shadow: 0 0 20px #0ff6;
            }
            
            .stButton button:hover {
                transform: scale(1.05);
                box-shadow: 0 0 40px #0ff9;
            }
            
            .stButton button:before {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: linear-gradient(45deg, transparent, #ffffff40, transparent);
                transform: rotate(45deg);
                animation: shine 3s infinite;
            }
            
            @keyframes shine {
                0% { transform: translateX(-100%) rotate(45deg); }
                100% { transform: translateX(100%) rotate(45deg); }
            }
            
            .prediction-text {
                font-size: 1.5rem;
                color: #0ff;
                text-align: center;
                margin: 15px 0;
            }
            
            .confidence-text {
                font-size: 2rem;
                color: #f0f;
                text-align: center;
                text-shadow: 0 0 10px #f0f;
            }
        </style>
        """, unsafe_allow_html=True
    )

    # App Header
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2917/2917996.png", width=80)
    with col2:
        st.markdown("<h1 class='cyberpunk'>PHYTOSCAN AI</h1>", unsafe_allow_html=True)
        st.markdown("<h3 class='cyberpunk' style='font-size: 1.5rem;'>NEURAL PLETHORA DIAGNOSIS MATRIX</h3>", unsafe_allow_html=True)

    st.markdown("---")

    # Holographic Upload Area
    st.markdown(
        """
        <div style="border: 2px dashed #0ff; border-radius: 15px; padding: 20px; margin: 20px 0; position: relative;">
            <div style="position: absolute; width: 100%; height: 100%; background: linear-gradient(45deg, #0ff2, #f0f2); mix-blend-mode: screen; pointer-events: none; animation: pulse 2s infinite;"></div>
            <div style="position: relative; z-index: 1;">
        """, unsafe_allow_html=True
    )
    
    uploaded_file = st.file_uploader(
        "🌌 UPLOAD PHYTO-TISSUE SAMPLE",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    st.markdown("</div></div>", unsafe_allow_html=True)

    if uploaded_file:
        # Image Preview
        with st.expander("🌀 HOLO-PREVIEW", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                img = Image.open(uploaded_file)
                st.image(img, caption="ORIGINAL SPECIMEN", use_column_width=True)
            with col2:
                processed_img = img.resize((224, 224))
                st.image(processed_img, caption="NEURAL INPUT MATRIX", use_column_width=True)

        # Prediction Trigger
        if st.button("🚀 INITIATE QUANTUM ANALYSIS", use_container_width=True):
            try:
                model = load_cached_model()

                with st.spinner("🌀 DECODING PHYTO-PATTERNS..."):
                    img_array = image.img_to_array(processed_img)
                    img_array = np.expand_dims(img_array, axis=0) / 255.0
                    predictions = model.predict(img_array)
                    predicted_class = CLASS_NAMES[np.argmax(predictions)]
                    confidence = np.max(predictions) * 100

                display_results(predicted_class, confidence)

            except Exception as e:
                st.error(f"QUANTUM COLLAPSE DETECTED: {str(e)}")

# ✅ Run App
if __name__ == "__main__":
    main()
