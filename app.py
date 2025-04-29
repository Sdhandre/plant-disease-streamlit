import streamlit as st
import numpy as np
import gdown
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ✅ Page Config with updated theme
st.set_page_config(
    page_title="PhytoScan AI - Plant Disease Detectimport streamlit as st
import numpy as np
import gdown
import os
import time
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

# ---------- Model Download ----------
MODEL_PATH = "plant_disease_modelfinal2.h5"
FILE_ID = "1tDt1NSWyfkqtFzh91KJQtPNVl5mbc2QG"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Download model if missing
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading AI model..."):
        gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=True)

# ---------- Cached Model Loader ----------
@st.cache_resource(show_spinner=False)
def load_cached_model():
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        raise

# ---------- UI Components ----------
def main():
    # Custom CSS (keep previous styles)
    
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
                st.image(img, caption="Original Image")
            with col2:
                processed_img = img.resize((224, 224))
                st.image(processed_img, caption="Processed Image")
        
        # Prediction Logic
        if st.button("🔍 Analyze Now", use_container_width=True):
            try:
                model = load_cached_model()
                with st.status("🧠 AI Analysis Progress", expanded=True) as status:
                    # Image preprocessing
                    st.write("Preprocessing image...")
                    img_array = image.img_to_array(processed_img)
                    img_array = np.expand_dims(img_array, axis=0) / 255.0
                    
                    # Prediction
                    st.write("Running diagnosis...")
                    predictions = model.predict(img_array)
                    predicted_class = CLASS_NAMES[np.argmax(predictions)]
                    confidence = np.max(predictions) * 100
                    
                    status.update(label="Analysis Complete", state="complete")
                
                # Display results
                display_results(predicted_class, confidence)
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

def display_results(predicted_class, confidence):
    # Keep previous display logic
    # ...

if __name__ == "__main__":
    main()ion",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ----------- Custom CSS -------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5ffe7 0%, #e6f4d7 100%);
    }
    
    .stApp {
        max-width: 850px;
        padding: 2rem;
    }
    
    .upload-section {
        border: 2px dashed #4CAF50;
        border-radius: 15px;
        padding: 2rem;
        background: rgba(76, 175, 80, 0.05);
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        background: rgba(76, 175, 80, 0.1);
        transform: translateY(-2px);
    }
    
    .prediction-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1.5rem 0;
    }
    
    .status-bar {
        padding: 0.8rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .healthy {
        background: #C8E6C9;
        color: #1B5E20;
    }
    
    .disease {
        background: #FFCDD2;
        color: #B71C1C;
    }
    
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
    
    button[data-baseweb="button"] {
        background: #4CAF50 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        transition: all 0.3s !important;
    }
    
    button[data-baseweb="button"]:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

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

# ----------- Header Section -------------
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2917/2917996.png", width=80)
with col2:
    st.title("PhytoScan AI")
    st.markdown("**AI-Powered Plant Health Diagnosis System**")

st.markdown("---")

# ----------- Main Content -------------
st.subheader("🌿 Upload Leaf Image")
with st.container():
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        " ",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption("Supports JPG, JPEG, PNG formats | Max size 5MB")

if uploaded_file is not None:
    with st.container():
        st.subheader("📸 Image Preview")
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            img = Image.open(uploaded_file)
            st.image(img, caption='Original Image', use_column_width=True)
        with col_img2:
            img_resized = img.resize((224, 224))
            st.image(img_resized, caption='Processed Image (224x224)', use_column_width=True)

    if st.button("🔍 Analyze Leaf Health", use_container_width=True):
        try:
            model = load_model_from_drive()
            with st.spinner("🧠 AI is analyzing..."):
                progress_bar = st.progress(0)
                
                # Simulate analysis steps
                for percent_complete in range(100):
                    progress_bar.progress(percent_complete + 1)
                    time.sleep(0.01)
                
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0

                prediction = model.predict(img_array)
                predicted_class = CLASS_NAMES[np.argmax(prediction)]
                confidence = np.max(prediction) * 100

                st.markdown("---")
                with st.container():
                    st.subheader("📋 Analysis Report")
                    
                    if "healthy" in predicted_class:
                        status_class = "healthy"
                        emoji = "✅"
                    else:
                        status_class = "disease"
                        emoji = "⚠️"
                    
                    st.markdown(f"""
                    <div class="status-bar {status_class}">
                        <h3>{emoji} {predicted_class.replace('___', ' ').replace('__', ' ')}</h3>
                        <p>Confidence: {confidence:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if status_class == "disease":
                        st.markdown("""
                        ## 🩺 Recommended Actions
                        - Immediately isolate affected plants
                        - Apply recommended fungicides
                        - Remove severely infected leaves
                        - Monitor plant health daily
                        - Consult agricultural expert
                        """)
                    else:
                        st.balloons()
                        st.markdown("""
                        ## 🌟 Healthy Plant Tips
                        - Maintain regular watering schedule
                        - Ensure proper sunlight exposure
                        - Monitor for early signs of pests
                        - Use organic fertilizers monthly
                        - Rotate crops seasonally
                        """)

        except Exception as e:
            st.error(f"⚠️ Analysis Error: {e}")
            
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>Powered by TensorFlow & Streamlit | 🌍 Sustainable Agriculture Initiative</p>
    <p>⚠️ For research purposes only | Consult experts for field diagnosis</p>
</div>
""", unsafe_allow_html=True)
