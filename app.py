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

# Theme switcher
dark_mode = st.sidebar.checkbox("Dark Mode", value=True)

# CSS for themes
if dark_mode:
    st.markdown(
        """
        <style> /* Dark theme CSS... */ </style>
        """, unsafe_allow_html=True)
else:
    st.markdown(
        """
        <style> /* Light theme CSS... */ </style>
        """, unsafe_allow_html=True)

# Disease info mapping
DISEASE_INFO = {
    # same mapping as before
}

# Constants
MODEL_PATH = "plant_disease_modelfinal2.h5"
FILE_ID = "1tDt1NSWyfkqtFzh91KJQtPNVl5mbc2QG"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"
CLASS_NAMES = list(DISEASE_INFO.keys())

# Model loader with caching
@st.cache_resource
def load_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model..."):
            gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=True)
    return load_model(MODEL_PATH)

# Load the model
try:
    model = load_model_from_drive()
    st.sidebar.success("Model loaded successfully")
except Exception as e:
    st.sidebar.error(f"Model load failed: {e}")

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation", ["Home", "Batch Prediction", "History", "Uploads Log", "About"]
)

# Header
st.markdown("<div class='title'>🍃 AgriScan</div>", unsafe_allow_html=True)
st.markdown("---")

if page == "Home":
    st.header("Single Image Prediction")
    upload_file = st.file_uploader("Upload a leaf image", type=["jpg","png","jpeg"])
    if upload_file:
        # Save the uploaded image
        uid = uuid.uuid4().hex
        save_path = os.path.join("uploads", f"{uid}_{upload_file.name}")
        with open(save_path, "wb") as f:
            f.write(upload_file.getbuffer())

        img = Image.open(upload_file).convert("RGB")
        st.image(img, use_column_width=True, caption="Leaf Image")
        if st.button("Predict Disease"):
            with st.spinner("Analyzing..."):
                arr = image.img_to_array(img.resize((224,224))) / 255.0
                preds = model.predict(np.expand_dims(arr,0))
                idx = int(np.argmax(preds))
                pred_key = CLASS_NAMES[idx]
                pred_label = pred_key.replace("_"," ")
                conf = float(np.max(preds))*100
            st.success(f"{pred_label} ({conf:.1f}%)")
            info = DISEASE_INFO[pred_key]
            with st.expander("Details"):
                st.markdown(f"**Cause:** {info['cause']}")
                st.markdown(f"**Prevention:** {info['prevention']}")
                st.markdown(f"**Treatment:** {info['treatment']}")
            # record history
            st.session_state.history.append({
                "timestamp": datetime.now(),
                "label": pred_label,
                "confidence": conf,
                "file": save_path
            })

elif page == "Batch Prediction":
    st.header("Batch Prediction")
    files = st.file_uploader("Upload multiple leaf images", type=["jpg","png","jpeg"], accept_multiple_files=True)
    if files:
        results = []
        for f in files:
            # Save each file
            uid = uuid.uuid4().hex
            path = os.path.join("uploads", f"{uid}_{f.name}")
            with open(path, "wb") as out:
                out.write(f.getbuffer())

            img = Image.open(f).convert("RGB")
            arr = image.img_to_array(img.resize((224,224))) / 255.0
            preds = model.predict(np.expand_dims(arr,0))
            idx = int(np.argmax(preds))
            pred_key = CLASS_NAMES[idx]
            label = pred_key.replace("_"," ")
            conf = float(np.max(preds))*100
            results.append({"Image": f.name, "Label": label, "Confidence": f"{conf:.1f}%"})
            st.session_state.history.append({"timestamp": datetime.now(), "label": label, "confidence": conf, "file": path})
        df = pd.DataFrame(results)
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", data=csv, file_name="batch_results.csv")

elif page == "History":
    st.header("Prediction History")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        hist_df['timestamp'] = hist_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(hist_df.drop(columns=['file']))
        csv = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download History as CSV", data=csv, file_name="history.csv")
    else:
        st.info("No history yet.")

elif page == "Uploads Log":
    st.header("Uploaded Images Log")
    files = os.listdir("uploads")
    if files:
        for file in sorted(files, reverse=True):
            path = os.path.join("uploads", file)
            with st.expander(file):
                st.image(path, use_column_width=True)
                with open(path, "rb") as img_file:
                    st.download_button("Download image", img_file, file, mime="image/png")
    else:
        st.info("No images uploaded yet.")

elif page == "About":
    st.header("About AgriScan")
    st.markdown(
        "AgriScan uses a CNN model to detect plant diseases. Users can view uploaded images, get diagnoses, and explore detailed remedies."
    )
    st.markdown("---")
    st.subheader("Features")
    st.write("• Single and batch predictions")
    st.write("• Uploads Log with image previews & downloads")
    st.write("• Prediction history with export")
    st.write("• Light/Dark theme toggle")
    st.write("• Detailed disease info (cause, prevention, treatment)")
    st.write("• CSV export of results")
    st.markdown("---")
    st.write("Developed by Your Name 🌱")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center;color:#999;'>&copy; 2025 AgriScan</p>", unsafe_allow_html=True)
