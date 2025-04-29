import streamlit as st
import numpy as np
import gdown
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# 🚀 Page Config
st.set_page_config(
    page_title="Agriscan - Next-Gen Plant Diagnosis",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🌟 Sidebar Controls
st.sidebar.markdown("## 🎛️ Settings")
theme = st.sidebar.selectbox("Theme", ["Cosmic", "Nebula Dreams", "Galactic Dawn"])
conf_threshold = st.sidebar.slider("Confidence Alert Threshold (%)", 50, 100, 75)
st.sidebar.markdown("---")
st.sidebar.markdown("🔗 [GitHub Repo](https://github.com/your_repo)")
st.sidebar.markdown("📖 [Documentation](https://your_docs.com)")
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Agriscan")

# 🌈 Dynamic CSS Injection
css = f'''
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
body, .css-1d391kg {{ font-family: 'Orbitron', sans-serif; }}
/* Background Themes */
{'' if theme=='Galactic Dawn' else theme=='Nebula Dreams' and 
"body {{ background: radial-gradient(circle at top left, #2b006b, #7000a1, #12002b); }}" or 
"body {{ background: radial-gradient(circle at 50% 50%, #0a0a1a, #000000); }}"}

/* Starfield */
body::before {{ content: ''; position: fixed; top:0; left:0; width:100%; height:100%;
 background: url('https://i.imgur.com/9bKZKfR.png') repeat; animation: moveStars 60s linear infinite;
 z-index:-1; opacity:0.15; }}
@keyframes moveStars {{ from{{transform:translate(0,0);}} to{{transform:translate(-2000px,1000px);}} }}

/* Shooting Stars */
body::after {{ content: ''; position: fixed; top: -10%; left: -10%; width: 200%; height: 200%;
 background: radial-gradient(2px 2px at 70% 20%, white, transparent),
 radial-gradient(1px 1px at 40% 60%, white, transparent);
 animation: shoot 5s linear infinite; opacity: 0.5; z-index:-1; }}
@keyframes shoot {{
 0% {{ background-position: 0% 0%, 100% 100%; }}
 100% {{ background-position: 100% 100%, 0% 0%; }}
}}

/* Glitch Title */
.stTitle h1 {{
  position: relative; color: #fff; font-size: 3rem; letter-spacing: 3px;
  animation: glitch 2s infinite;
}}
@keyframes glitch {{
  0% {{ text-shadow: 2px 2px #f0f, -2px -2px #0ff; }}
  50% {{ text-shadow: -2px 2px #0ff, 2px -2px #f0f; }}
  100% {{ text-shadow: 2px 2px #f0f, -2px -2px #0ff; }}
}}

/* Buttons */
.stButton>button {{
  background: linear-gradient(45deg, #ff4ecb, #4e6cff, #00ffe7);
  background-size: 200% 200%; animation: gradient 5s ease infinite;
  border:none; border-radius: 12px; padding: 0.8em 1.6em;
  box-shadow: 0 0 20px rgba(78, 110, 255, 0.8);
  transition: transform 0.2s;
}}
.stButton>button:hover {{ transform: scale(1.05) rotate(1deg); }}
@keyframes gradient {{ 0%{{background-position:0 50%;}}50%{{background-position:100% 50%;}}100%{{background-position:0 50%;}} }}

/* Result Card */
.result-card {{ background: rgba(20,20,50,0.9); padding: 25px; border-radius: 20px;
 box-shadow: 0 0 30px rgba(78,203,255,0.7);
}}

/* Image Frames */
.stImage img {{ border: 4px solid #4e6cff; border-radius: 16px; }}
</style>
'''
st.markdown(css, unsafe_allow_html=True)

# 🔧 Model Setup
MODEL_PATH = 'plant_disease_modelfinal2.h5'
FILE_ID = '1tDt1NSWyfkqtFzh91KJQtPNVl5mbc2QG'
URL = f'https://drive.google.com/uc?id={FILE_ID}'
if not os.path.exists(MODEL_PATH):
    with st.spinner('Downloading AI model...'):
        gdown.download(URL, MODEL_PATH, quiet=True)
model = load_model(MODEL_PATH)

CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

# 🪴 App Logic
def main():
    # Header
    st.markdown('<div class="stTitle"><h1>Agriscan</h1></div>', unsafe_allow_html=True)
    st.markdown('**AI-Powered Plant Health Diagnostics**')
    st.markdown('---')

    # Upload & Preview
    st.subheader('🌿 Upload Leaf Image')
    uploaded = st.file_uploader('', type=['jpg','jpeg','png'])
    if not uploaded:
        st.info('Awaiting image upload...')
        return
    img = Image.open(uploaded)
    resized = img.resize((224,224))
    with st.expander('📸 Preview', expanded=True):
        c1, c2 = st.columns(2)
        c1.image(img, caption='Original', use_column_width=True)
        c2.image(resized, caption='Resized', use_column_width=True)

    # Analyze
    if st.button('🔍 Analyze Now'):
        progress = st.progress(0)
        for pct in range(0, 101, 20):
            progress.progress(pct)
            st.sleep(0.1)
        arr = image.img_to_array(resized)/255.0
        preds = model.predict(np.expand_dims(arr,0))
        idx = np.argmax(preds)
        cls = CLASS_NAMES[idx].replace('_',' ')
        conf = preds[0][idx]*100

        # Highlight low confidence
        if conf < conf_threshold:
            st.warning(f"⚠️ Low confidence: {conf:.2f}%")

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"<h2>Result: {cls}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h3>Confidence: {conf:.2f}%</h3>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # About
    with st.expander('ℹ️ About'):
        st.write('Agriscan v2.0 - the leading cosmic plant health diagnosis tool. Built with ❤️ by Agriscan Team.')

if __name__=='__main__':
    main()
