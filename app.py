import streamlit as st
import numpy as np
import gdown
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# 🚀 Page Config
st.set_page_config(
    page_title="PhytoScan AI - Plant Disease Detection",
    page_icon="🌌",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ✨ Inject Cosmic Styles
st.markdown(
    '''
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');

    /* Main container styling */
    .reportview-container .main .block-container {
        position: relative;
        background: radial-gradient(circle at 50% 50%, #0a0a1a, #000000) !important;
        z-index: 0;
        font-family: 'Orbitron', sans-serif;
        color: #e0e0ff;
    }
    /* Starfield Overlay */
    .reportview-container .main .block-container:before {
        content: "";
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: url('https://i.imgur.com/9bKZKfR.png') repeat;
        animation: starMove 60s linear infinite;
        opacity: 0.2;
        z-index: -1;
    }
    @keyframes starMove {
        from { background-position: 0 0; }
        to   { background-position: -10000px 5000px; }
    }
    /* Button glow animation */
    .stButton>button {
        background: linear-gradient(45deg, #ff4ecb, #4e6cff, #00ffe7);
        background-size: 200% 200%;
        animation: gradientBG 8s ease infinite;
        border: none;
        border-radius: 12px;
        padding: 0.8em 1.6em;
        box-shadow: 0 0 15px rgba(78, 110, 255, 0.7);
        font-weight: bold;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    /* Diagnosis card styling */
    .stMarkdown div[style*="Diagnosis Result"] {
        background: rgba(20, 20, 50, 0.8);
        border: 2px solid #4ecbff;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 0 20px rgba(78, 203, 255, 0.8);
        animation: cardGlow 2s ease-in-out infinite alternate;
    }
    @keyframes cardGlow {
        from { box-shadow: 0 0 10px #4ecbff; }
        to   { box-shadow: 0 0 30px #ff4ecb; }
    }
    /* Image frame styling */
    .stImage img {
        border: 3px solid #4e6cff;
        border-radius: 12px;
    }
    </style>
    ''', unsafe_allow_html=True
)

# 🔧 Model Paths & Classes
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

# ✅ Download model if missing
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI model..."):
            gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=True)

download_model()

# 🛠 Cached Model Loader
@st.cache_resource(show_spinner=False)
def load_cached_model():
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        raise

# 🃏 Display Results
def display_results(predicted_class, confidence):
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style=\"text-align:center;\">
            <div style=\"background: rgba(20,20,50,0.8); padding:20px; border-radius:15px; display:inline-block;\">
                <h3>Diagnosis Result</h3>
                <h4><strong>Predicted:</strong> {predicted_class.replace('_',' ')}</h4>
                <h5><strong>Confidence:</strong> {confidence:.2f}%</h5>
            </div>
        </div>
        """, unsafe_allow_html=True
    )

# 🪴 Main App
def main():
    # Header
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2917/2917996.png", width=80)
    with col2:
        st.title("PhytoScan AI")
        st.markdown("**AI-Powered Plant Health Diagnosis System**")

    st.markdown("---")

    # Upload Section
    st.subheader("🌿 Upload a Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg","jpeg","png"], label_visibility="collapsed"
    )

    if uploaded_file:
        img = Image.open(uploaded_file)
        processed_img = img.resize((224,224))

        with st.expander("📸 Preview", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.image(img, caption="Original", use_column_width=True)
            with c2:
                st.image(processed_img, caption="Resized (224x224)", use_column_width=True)

        if st.button("🔍 Analyze Now", use_container_width=True):
            try:
                model = load_cached_model()
                with st.spinner("🧠 AI Analysis in Progress..."):
                    img_arr = image.img_to_array(processed_img) / 255.0
                    img_arr = np.expand_dims(img_arr, axis=0)

                    preds = model.predict(img_arr)
                    pred_class = CLASS_NAMES[np.argmax(preds)]
                    conf = np.max(preds) * 100

                display_results(pred_class, conf)
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()
