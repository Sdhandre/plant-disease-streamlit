import streamlit as st
import numpy as np
import gdown
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="AgriScan-Plant Disease Detector",
    page_icon="🍃",
    layout="wide"
)

# Custom CSS for Dark Theme
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

# Disease info mapping
DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "cause": "Bacterial spot is caused by the bacterium Xanthomonas euvesicatoria, which infects pepper leaves and fruits.",
        "prevention": "Use disease-free seeds, practice crop rotation, remove crop debris, and avoid overhead irrigation.",
        "treatment": "Apply copper-based bactericides and follow integrated pest management practices to reduce spread."
    },
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

# Constants
MODEL_PATH = "plant_disease_modelfinal2.h5"
FILE_ID = "1tDt1NSWyfkqtFzh91KJQtPNVl5mbc2QG"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"
CLASS_NAMES = list(DISEASE_INFO.keys())

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
st.markdown("<div class='title'>🌿 AgriScan - Plant Disease Detection</div>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("About AgriScan")
    st.write("Upload a leaf image and get instant diagnosis, with detailed cause, prevention, and treatment tips.")
    st.markdown("---")
    st.subheader("Disease Classes")
    for cls in CLASS_NAMES:
        st.write(f"• {cls.replace('_', ' ')}")
    st.markdown("---")
    st.write("## Tips for Healthy Plants")
    st.write("• Rotate crops regularly")
    st.write("• Ensure proper drainage and soil health")
    st.write("• Monitor for pests and diseases weekly")
    st.markdown("---")
    st.write("Developed by Your Name 🌱")

# Main layout
upload_col, result_col = st.columns(2)

with upload_col:
    st.subheader("1. Upload Leaf Image")
    uploaded_file = st.file_uploader(
        label="", type=["jpg", "jpeg", "png"], help="Select a clear leaf image"
    )

with result_col:
    st.subheader("2. Prediction & Details")
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
                pred_key = CLASS_NAMES[pred_idx]
                pred_label = pred_key.replace("_", " ")
                confidence = float(np.max(preds)) * 100
            st.success(f"**{pred_label}** detected with {confidence:.2f}% confidence! 🌿")

            # Disease details expander
            info = DISEASE_INFO[pred_key]
            with st.expander("Learn more about this disease"):
                st.markdown(f"### Cause\n{info['cause']}")
                st.markdown(f"### Prevention\n{info['prevention']}")
                st.markdown(f"### Treatment\n{info['treatment']}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    elif not uploaded_file:
        st.info("Please upload an image to begin.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #999;'>&copy; 2025 AgriScan | Powered by TensorFlow & Streamlit</p>",
    unsafe_allow_html=True
)
