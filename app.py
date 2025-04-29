import streamlit as st
import numpy as np
import gdown
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AgriScan - Plant Disease Detector",
    page_icon="🍃",
    layout="wide"
)

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Theme switcher
dark_mode = st.sidebar.checkbox("Dark Mode", value=True)

# CSS for themes
if dark_mode:
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(135deg, #121212 0%, #1e1e1e 100%); }
        html, body, [class^="css"] { color: #e0e0e0 !important; background-color: transparent; }
        .title { font-size: 3rem; font-weight: bold; text-align: center; color: #80cbc4; margin-top: 1rem; }
        .stMarkdown h2, .stMarkdown h3 { color: #4dd0e1 !important; font-weight: bold; }
        div.stButton > button { background-color: #26a69a; color: white !important; font-size: 1.2rem; padding: 0.6rem 1.2rem; border-radius: 12px; border: none; }
        div.stButton > button:hover { background-color: #00796b; }
        .css-6qob1r, .css-1d391kg, .css-1v3fvcr { background-color: #1e1e1e !important; color: #e0e0e0 !important; }
        .stFileUploader, .stTextInput, .stSelectbox, .stSlider, .stNumberInput { background-color: #2c2c2c !important; color: #e0e0e0 !important; }
        footer, footer * { color: #999 !important; background: transparent; }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        .stApp { background: #f7f7f7; }
        html, body, [class^="css"] { color: #333 !important; background-color: transparent; }
        .title { font-size: 3rem; font-weight: bold; text-align: center; color: #2e7d32; margin-top: 1rem; }
        .stMarkdown h2, .stMarkdown h3 { color: #388e3c !important; font-weight: bold; }
        div.stButton > button { background-color: #66bb6a; color: white !important; font-size: 1.2rem; padding: 0.6rem 1.2rem; border-radius: 12px; border: none; }
        div.stButton > button:hover { background-color: #4caf50; }
        .css-6qob1r, .css-1d391kg, .css-1v3fvcr { background-color: #ffffff !important; color: #333 !important; }
        .stFileUploader, .stTextInput, .stSelectbox, .stSlider, .stNumberInput { background-color: #f0f0f0 !important; color: #333 !important; }
        footer, footer * { color: #555 !important; background: transparent; }
        </style>
        """,
        unsafe_allow_html=True
    )

# Disease info mapping
di = st.experimental_singleton(lambda: {
    "Pepper__bell___Bacterial_spot": {
        "cause": "Bacterial spot is caused by the bacterium Xanthomonas euvesicatoria.",
        "prevention": "Use disease-free seeds, rotate crops, remove debris.",
        "treatment": "Apply copper-based bactericides and IPM."
    },
    # ... other diseases ...
    "Pepper__bell___healthy": {
        "cause": "No disease detected. The plant appears healthy.",
        "prevention": "Maintain regular watering, proper fertilization, and monitor for pests.",
        "treatment": "Not applicable. Continue good agricultural practices."
    },
    "Potato___Early_blight": {
        "cause": "Early blight is caused by the fungus Alternaria solani which produces lesions on leaves and stems.",
        "prevention": "Rotate crops, remove volunteer potato plants, and mulch to prevent soil contact.",
        "treatment": "Use fungicides containing chlorothalonil or mancozeb and remove infected foliage."
    },
    "Potato___Late_blight": {
        "cause": "Late blight is caused by the oomycete Phytophthora infestans, leading to dark lesions and tuber rot.",
        "prevention": "Avoid overhead watering, ensure good air circulation, and destroy infected plants.",
        "treatment": "Apply fungicides such as mancozeb and practice resistant variety planting."
    },
    "Potato___healthy": {
        "cause": "No disease detected. The plant appears healthy.",
        "prevention": "Maintain proper soil moisture, fertilization, and inspect regularly for pests.",
        "treatment": "Not applicable. Continue good agricultural practices."
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "cause": "Tomato Yellow Leaf Curl Virus (TYLCV) is transmitted by whiteflies causing leaf curl and yellowing.",
        "prevention": "Use resistant varieties, control whitefly populations, and employ reflective mulches.",
        "treatment": "No cure; remove and destroy infected plants, and prevent spread by controlling vectors."
    },
    "Tomato__Tomato_mosaic_virus": {
        "cause": "Tomato Mosaic Virus (ToMV) spreads via contaminated tools, hands, and seeds, causing mottling.",
        "prevention": "Sanitize tools, use certified seeds, and isolate infected plants.",
        "treatment": "No chemical cure; rogue out infected plants and maintain strict sanitation."
    },
    "Tomato_healthy": {
        "cause": "No disease detected. The plant appears healthy.",
        "prevention": "Ensure balanced nutrients, consistent watering, and monitor pests regularly.",
        "treatment": "Not applicable. Continue good agricultural practices."
    }
}
)

# Constants
MODEL_PATH = "plant_disease_modelfinal2.h5"
FILE_ID = "1tDt1NSWyfkqtFzh91KJQtPNVl5mbc2QG"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"
CLASS_NAMES = list(di.keys())

# Model loader with caching
@st.cache_resource
def load_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model..."):
            gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=True)
    return load_model(MODEL_PATH)

model = None
try:
    model = load_model_from_drive()
    st.sidebar.success("Model loaded successfully")
except Exception as e:
    st.sidebar.error(f"Model load failed: {e}")

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation", ["Home", "Batch Prediction", "History", "About"]
)

# Header
st.markdown("<div class='title'>🍃 AgriScan</div>", unsafe_allow_html=True)
st.markdown("---")

if page == "Home":
    st.header("Single Image Prediction")
    upload_file = st.file_uploader("Upload a leaf image", type=["jpg","png","jpeg"])
    if upload_file:
        img = Image.open(upload_file).convert("RGB")
        st.image(img, use_column_width=True, caption="Leaf Image")
        if st.button("Predict Disease"):
            cols = st.columns(2)
            with cols[1]:
                with st.spinner("Analyzing..."):
                    arr = image.img_to_array(img.resize((224,224))) / 255.0
                    pred = model.predict(np.expand_dims(arr,0))
                    idx = int(np.argmax(pred))
                    label = CLASS_NAMES[idx].replace("_"," ")
                    conf = float(np.max(pred))*100
                st.success(f"{label} ({conf:.1f}%)")
                info = di[label.replace(' ', '_')]
                with st.expander("Details"):
                    st.markdown(f"**Cause:** {info['cause']}")
                    st.markdown(f"**Prevention:** {info['prevention']}")
                    st.markdown(f"**Treatment:** {info['treatment']}")
                # record history
                st.session_state.history.append({
                    "timestamp": datetime.now(),
                    "label": label,
                    "confidence": conf
                })

elif page == "Batch Prediction":
    st.header("Batch Prediction")
    files = st.file_uploader("Upload multiple leaf images", type=["jpg","png","jpeg"], accept_multiple_files=True)
    if files:
        results = []
        for f in files:
            img = Image.open(f).convert("RGB"); arr = image.img_to_array(img.resize((224,224))) / 255.0
            pred = model.predict(np.expand_dims(arr,0))
            idx = int(np.argmax(pred))
            label = CLASS_NAMES[idx].replace("_"," ")
            conf = float(np.max(pred))*100
            results.append({"Image": f.name, "Label": label, "Confidence": f"{conf:.1f}%"})
            st.session_state.history.append({"timestamp": datetime.now(), "label": label, "confidence": conf})
        df = pd.DataFrame(results)
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", data=csv, file_name="batch_results.csv")

elif page == "History":
    st.header("Prediction History")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        hist_df['timestamp'] = hist_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(hist_df)
        csv = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download History as CSV", data=csv, file_name="history.csv")
    else:
        st.info("No history yet.")

elif page == "About":
    st.header("About AgriScan")
    st.markdown(
        "AgriScan uses a CNN model to detect common plant diseases. Upload leaf images to get instant insights on cause, prevention, and treatment."
    )
    st.markdown("---")
    st.subheader("Features")
    st.write("• Single and batch predictions")
    st.write("• Prediction history with export")
    st.write("• Light/Dark theme toggle")
    st.write("• Detailed disease info (cause, prevention, treatment)")
    st.write("• CSV export of results")
    st.markdown("---")
    st.write("Developed by Your Name 🌱")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center;color:#999;'>&copy; 2025 AgriScan</p>", unsafe_allow_html=True)

