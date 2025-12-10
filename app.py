import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DermaVision AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. HIGH-CONTRAST MEDICAL THEME (CSS) ---
st.markdown("""
<style>
    /* FORCE LIGHT MODE AESTHETIC */
    .stApp {
        background-color: #f0f4f8; /* Soft Clinical Blue-Grey */
    }
    
    /* FORCE TEXT COLOR (Fixes the invisible text issue) */
    h1, h2, h3, h4, h5, h6, p, li, span, div {
        color: #1e293b !important; /* Dark Slate Blue */
    }
    
    /* SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 2px solid #e2e8f0;
    }
    
    /* RESULT CARD STYLING */
    .diagnosis-card {
        background-color: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-left: 8px solid #0ea5e9; /* Light Blue Accent */
        text-align: center;
    }
    
    /* BUTTON STYLING */
    div.stButton > button {
        background: linear-gradient(to right, #0ea5e9, #2563eb);
        color: white !important;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: bold;
        transition: transform 0.2s;
        width: 100%;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        color: white !important;
    }
    
    /* UPLOAD BOX STYLING */
    [data-testid="stFileUploader"] {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border: 2px dashed #cbd5e1;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_derma_model():
    model_path = 'models/best_skin_model.keras'
    if not os.path.exists(model_path):
        st.error(f"❌ critical Error: Model file missing at `{model_path}`")
        st.info("👉 Please ensure you have a folder named 'models' containing 'best_skin_model.keras'")
        st.stop()
    return tf.keras.models.load_model(model_path)

try:
    model = load_derma_model()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# --- 4. DATA DEFINITIONS (HAM10000) ---
classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
class_details = {
    'akiec': ('Actinic Keratoses', 'Pre-cancerous'),
    'bcc': ('Basal Cell Carcinoma', 'Cancerous'),
    'bkl': ('Benign Keratosis', 'Benign'),
    'df': ('Dermatofibroma', 'Benign'),
    'mel': ('Melanoma', 'Cancerous (High Risk)'),
    'nv': ('Melanocytic Nevi', 'Benign (Mole)'),
    'vasc': ('Vascular Lesion', 'Benign')
}

# --- 5. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=70)
    st.title("Patient Metadata")
    st.markdown("---")
    
    st.info("ℹ️ **Why do we need this?**\nSkin lesions look different on different body parts and age groups. This data helps the AI be more accurate.")
    
    age = st.slider("Patient Age", 0, 100, 30)
    sex = st.radio("Biological Sex", ["Male", "Female", "Unknown"], horizontal=True)
    
    # Accurate HAM10000 Locations
    loc = st.selectbox("Anatomical Site", [
        "Back", "Lower Extremity", "Trunk", "Upper Extremity", "Abdomen", 
        "Face", "Chest", "Foot", "Neck", "Scalp", "Hand", "Ear", 
        "Genital", "Acral", "Unknown"
    ])

# --- 6. HELPER FUNCTIONS ---
def build_meta_vector(age, sex, loc):
    # Matches Training Logic exactly
    sex_v = [0, 0, 0]
    if sex == 'Female': sex_v[0] = 1
    elif sex == 'Male': sex_v[1] = 1
    else: sex_v[2] = 1
    
    locs = ["abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital", 
            "hand", "lower extremity", "neck", "scalp", "trunk", "upper extremity", "unknown"]
    
    loc_v = [0] * len(locs)
    if loc.lower() in locs:
        loc_v[locs.index(loc.lower())] = 1
        
    return np.array(sex_v + loc_v + [age / 100.0]).reshape(1, -1)

# --- 7. MAIN INTERFACE ---
col_logo, col_title = st.columns([1, 5])
with col_title:
    st.title("DermaVision AI")
    st.markdown("**Professional Dermatoscopic Analysis System**")

# Beginners Guide (Collapsible)
with st.expander("📖 New User Guide: How to use this tool", expanded=True):
    st.markdown("""
    1. **Enter Patient Data:** Use the sidebar on the left to set Age, Sex, and Location.
    2. **Upload Image:** Take a clear, close-up photo of the skin lesion and upload it below.
    3. **Analyze:** Click the 'Analyze Lesion' button.
    4. **Review:** Check the diagnosis and confidence score.
    """)

# Main Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Image Acquisition")
    uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Clinical Specimen", use_column_width=True)
        
        # Quality Warning
        if image.size[0] < 224:
            st.warning("⚠️ Image resolution is low. Results may be less accurate.")

with col2:
    st.subheader("2. Diagnostic Engine")
    
    if uploaded_file:
        if st.button("🔍 Analyze Lesion Now"):
            with st.spinner("Analyzing cell patterns & metadata..."):
                # Preprocessing
                img_resized = image.resize((224, 224))
                img_array = preprocess_input(np.array(img_resized))
                img_batch = np.expand_dims(img_array, axis=0)
                meta_batch = build_meta_vector(age, sex, loc)
                
                # Prediction
                try:
                    preds = model.predict({'image_input': img_batch, 'meta_input': meta_batch})
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
                    st.stop()
                
                # Results
                top_idx = np.argmax(preds)
                label = classes[top_idx]
                full_name, status = class_details[label]
                conf = np.max(preds)
                
                # Color Logic
                color_code = "#e74c3c" if "Cancer" in status else "#27ae60"
                
                # BEAUTIFUL RESULT CARD
                st.markdown(f"""
                <div class="diagnosis-card" style="border-left: 8px solid {color_code};">
                    <h3 style="margin:0; color:#64748b;">Primary Diagnosis</h3>
                    <h1 style="margin:10px 0; color:#1e293b;">{full_name}</h1>
                    <p style="font-size:20px; font-weight:bold; color:{color_code};">{status}</p>
                    <hr>
                    <p style="color:#64748b; margin-bottom:0;">AI Confidence Score</p>
                    <h2 style="color:#3b82f6; margin:0;">{conf*100:.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Detailed Chart
                st.write("")
                st.markdown("#### Probability Breakdown")
                chart_data = pd.DataFrame({
                    "Condition": [class_details[c][0] for c in classes],
                    "Probability": preds[0]
                })
                st.bar_chart(chart_data.set_index("Condition"))

    else:
        # Empty State
        st.info("👈 Please upload an image on the left to activate the AI.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 12px;">
    <strong>Medical Disclaimer:</strong> This tool is for educational purposes only. 
    AI predictions should never replace professional medical advice. 
    Always consult a dermatologist for diagnosis.
</div>
""", unsafe_allow_html=True)
