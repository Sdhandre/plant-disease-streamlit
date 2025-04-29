import streamlit as st
import numpy as np
import gdown
import os
import uuid
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import pandas as pd
from datetime import datetime

# Ensure upload directory exists
os.makedirs("uploads", exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="AgriScan - Plant Disease Detector",
    page_icon="🍃",
    layout="wide"
)

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Sidebar settings only
with st.sidebar:
    # Theme toggle
    dark_mode = st.checkbox("Dark Mode", value=True)
    st.markdown("---")
    st.write("**Settings**")

# Apply theme CSS
if dark_mode:
    st.markdown(
        '''
        <style>
        .stApp { background: linear-gradient(135deg, #121212 0%, #1e1e1e 100%); }
        html, body, [class^="css"] { color: #e0e0e0 !important; background-color: transparent; }
        .title { font-size: 3rem; font-weight: bold; text-align: center; color: #80cbc4; margin-top: 1rem; }
        </style>
        ''',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '''
        <style>
        .stApp { background: #f7f7f7; }
        html, body, [class^="css"] { color: #333 !important; background-color: transparent; }
        .title { font-size: 3rem; font-weight: bold; text-align: center; color: #2e7d32; margin-top: 1rem; }
        </style>
        ''',
        unsafe_allow_html=True
    )

# Disease info mapping
DISEASE_INFO = {
    'Pepper__bell___Bacterial_spot': {
        'cause': 'Bacterial spot is caused by Xanthomonas euvesicatoria.',
        'prevention': 'Use disease-free seeds, rotate crops, remove debris.',
        'treatment': 'Apply copper-based bactericides and IPM practices.'
    },
    'Pepper__bell___healthy': {
        'cause': 'No disease detected. The plant appears healthy.',
        'prevention': 'Maintain regular watering and proper fertilization.',
        'treatment': 'Not applicable.'
    },
    'Potato___Early_blight': {
        'cause': 'Early blight is caused by Alternaria solani.',
        'prevention': 'Rotate crops, remove volunteer plants, mulch.',
        'treatment': 'Use fungicides with chlorothalonil or mancozeb.'
    },
    'Potato___Late_blight': {
        'cause': 'Late blight is caused by Phytophthora infestans.',
        'prevention': 'Improve air circulation, destroy infected plants.',
        'treatment': 'Apply mancozeb, plant resistant varieties.'
    },
    'Potato___healthy': {
        'cause': 'No disease detected. The plant appears healthy.',
        'prevention': 'Maintain soil moisture and inspect regularly.',
        'treatment': 'Not applicable.'
    },
    'Tomato__Tomato_YellowLeaf__Curl_Virus': {
        'cause': 'TYLCV is transmitted by whiteflies.',
        'prevention': 'Use resistant varieties, control vectors.',
        'treatment': 'Remove infected plants; no cure.'
    },
    'Tomato__Tomato_mosaic_virus': {
        'cause': 'ToMV spreads via tools, hands, seeds.',
        'prevention': 'Sanitize tools, use certified seeds.',
        'treatment': 'Remove infected plants; maintain sanitation.'
    },
    'Tomato_healthy': {
        'cause': 'No disease detected. The plant appears healthy.',
        'prevention': 'Ensure nutrients and monitor pests.',
        'treatment': 'Not applicable.'
    }
}
CLASS_NAMES = list(DISEASE_INFO.keys())

# Model loading\ nMODEL_PATH = "plant_disease_modelfinal2.h5"
FILE_ID = "1tDt1NSWyfkqtFzh91KJQtPNVl5mbc2QG"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

@st.cache_resource
def load_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model..."):
            gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=True)
    return load_model(MODEL_PATH)

# Load model
try:
    model = load_model_from_drive()
    st.sidebar.success("Model loaded")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

# Header
st.markdown("<div class='title'>🍃 AgriScan</div>", unsafe_allow_html=True)
st.markdown("---")

# Top-level tabs for navigation (mobile-friendly)
tab_home, tab_batch, tab_history, tab_uploads, tab_about = st.tabs([
    "Home", "Batch", "History", "Uploads", "About"
])

# Home tab
with tab_home:
    st.header("Single Image Prediction")
    upload_file = st.file_uploader("Upload a leaf image", type=["jpg","png","jpeg"])
    if upload_file:
        uid = uuid.uuid4().hex
        save_path = os.path.join("uploads", f"{uid}_{upload_file.name}")
        with open(save_path, "wb") as f:
            f.write(upload_file.getbuffer())
        img = Image.open(save_path).convert("RGB")
        st.image(img, use_column_width=True, caption="Leaf Image")
        if st.button("Predict Disease"):
            with st.spinner("Analyzing..."):
                arr = image.img_to_array(img.resize((224,224))) / 255.0
                preds = model.predict(np.expand_dims(arr,0))
                idx = int(np.argmax(preds))
                key = CLASS_NAMES[idx]
                label = key.replace("_"," ")
                conf = float(np.max(preds)) * 100
            st.success(f"{label} ({conf:.1f}%)")
            info = DISEASE_INFO[key]
            with st.expander("Details"):
                st.markdown(f"**Cause:** {info['cause']}")
                st.markdown(f"**Prevention:** {info['prevention']}")
                st.markdown(f"**Treatment:** {info['treatment']}")
            st.session_state.history.append({
                "timestamp": datetime.now(),
                "label": label,
                "confidence": conf,
                "file": save_path
            })

# Batch tab
with tab_batch:
    st.header("Batch Prediction")
    files = st.file_uploader("Upload multiple images", type=["jpg","png","jpeg"], accept_multiple_files=True)
    if files:
        results = []
        for f in files:
            uid = uuid.uuid4().hex
            path = os.path.join("uploads", f"{uid}_{f.name}")
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            img = Image.open(path).convert("RGB")
            arr = image.img_to_array(img.resize((224,224))) / 255.0
            preds = model.predict(np.expand_dims(arr,0))
            idx = int(np.argmax(preds))
            key = CLASS_NAMES[idx]
            label = key.replace("_"," ")
            conf = float(np.max(preds)) * 100
            results.append({"Image": f.name, "Label": label, "Confidence": f"{conf:.1f}%"})
            st.session_state.history.append({"timestamp": datetime.now(), "label": label, "confidence": conf, "file": path})
        df = pd.DataFrame(results)
        st.dataframe(df)
        st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), file_name="batch.csv")

# History tab
with tab_history:
    st.header("Prediction History")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        hist_df['timestamp'] = hist_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(hist_df.drop(columns=['file']))
        st.download_button("Download History", hist_df.to_csv(index=False).encode('utf-8'), file_name="history.csv")
    else:
        st.info("No history yet.")

# Uploads tab
with tab_uploads:
    st.header("Uploaded Images Log")
    files = os.listdir("uploads")
    if files:
        for fname in sorted(files, reverse=True):
            path = os.path.join("uploads", fname)
            with st.expander(fname):
                st.image(path, use_column_width=True)
                with open(path, "rb") as img_file:
                    st.download_button("Download", img_file, fname, mime="image/png")
    else:
        st.info("No uploads yet.")

# About tab
with tab_about:
    st.header("About AgriScan")
    st.markdown("AgriScan uses a CNN to detect plant diseases. Browse tabs above for features.")
    st.markdown("---")
    st.write("**Features:**")
    st.write("- Single & batch predictions")
    st.write("- Uploads log with previews & downloads")
    st.write("- History export")
    st.write("- Light/Dark mode")
    st.write("- Detailed disease info")
    st.markdown("---")
    st.write("Developed by Your Name 🌱")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center;color:#999;'>&copy; 2025 AgriScan</p>", unsafe_allow_html=True)
