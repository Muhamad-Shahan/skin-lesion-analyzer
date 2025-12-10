import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DermaVision Pro",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. THEME ENFORCEMENT (The Fix) ---
st.markdown("""
<style>
    /* 1. Force entire app background to Light Grey */
    .stApp {
        background-color: #f4f6f9;
    }
    
    /* 2. FORCE ALL TEXT TO BE DARK (Overrides Dark Mode) */
    h1, h2, h3, h4, h5, h6, p, li, label, .stMarkdown, .stText {
        color: #1e293b !important; /* Dark Slate Blue */
    }
    
    /* 3. Fix Input Box Visibility */
    .stSelectbox div[data-baseweb="select"] > div,
    .stNumberInput div[data-baseweb="input"] > div,
    .stTextInput div[data-baseweb="input"] > div {
        background-color: white !important;
        color: black !important;
        border: 1px solid #cbd5e1;
    }
    
    /* Force text inside inputs to be black */
    input {
        color: black !important;
    }
    
    /* 4. Sidebar Background */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 2px solid #e2e8f0;
    }
    
    /* 5. Result Card */
    .diagnosis-card {
        background-color: white;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    
    /* 6. Button Styling */
    div.stButton > button {
        background: linear-gradient(135deg, #0284c7, #0369a1);
        color: white !important;
        border: none;
        padding: 14px;
        font-size: 16px;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(2, 132, 199, 0.3);
    }
    
    /* 7. Fix Expander Text */
    .streamlit-expanderHeader {
        color: #1e293b !important;
        background-color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_derma_model():
    model_path = 'models/best_skin_model.keras'
    if not os.path.exists(model_path):
        st.error(f"❌ System Error: Model missing at `{model_path}`")
        st.stop()
    return tf.keras.models.load_model(model_path)

try:
    model = load_derma_model()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# --- 4. DATA STANDARDS (HAM10000) ---
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

# --- 5. SIDEBAR (Patient Intake) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=70)
    st.title("Clinical Intake")
    st.markdown("---")
    
    # Tooltips added for "Beginner Friendliness"
    age = st.number_input("Patient Age", min_value=0, max_value=120, value=45, help="Age is a key factor in skin cancer risk.")
    sex = st.selectbox("Biological Sex", ["Male", "Female", "Unknown"])
    
    loc = st.selectbox("Anatomical Site", [
        "Back", "Lower Extremity", "Trunk", "Upper Extremity", "Abdomen", 
        "Face", "Chest", "Foot", "Neck", "Scalp", "Hand", "Ear", 
        "Genital", "Acral", "Unknown"
    ], help="Select the exact location where the image was taken.")
    
    st.markdown("---")
    st.info("ℹ️ **Privacy Note:** No patient data is stored locally. All processing happens in temporary memory.")

# --- 6. HELPER FUNCTIONS ---
def build_meta_vector(age, sex, loc):
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
    st.title("DermaVision Pro")
    st.markdown("**AI-Assisted Dermatoscopy Analysis System**")

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
    st.subheader("1. Specimen Acquisition")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], label_visibility="visible")
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Clinical Specimen", use_column_width=True)
        
        if image.size[0] < 224:
            st.warning("⚠️ Low Resolution Warning")

with col2:
    st.subheader("2. Diagnostic Engine")
    
    if uploaded_file:
        if st.button("RUN DIAGNOSTIC ANALYSIS"):
            with st.spinner("Analyzing cellular patterns..."):
                # Preprocessing
                img_resized = image.resize((224, 224))
                img_array = preprocess_input(np.array(img_resized))
                img_batch = np.expand_dims(img_array, axis=0)
                meta_batch = build_meta_vector(age, sex, loc)
                
                try:
                    preds = model.predict({'image_input': img_batch, 'meta_input': meta_batch})
                except Exception as e:
                    st.error(f"Shape Error: {e}")
                    st.stop()
                
                # Results
                top_idx = np.argmax(preds)
                label = classes[top_idx]
                full_name, status = class_details[label]
                conf = np.max(preds)
                
                # Color Logic
                if "Cancer" in status:
                    border_color = "#e74c3c" # Red
                    bg_color = "#fdf2f2" # Light Red BG
                    icon = "🚨"
                else:
                    border_color = "#27ae60" # Green
                    bg_color = "#f0fdf4" # Light Green BG
                    icon = "✅"
                
                # BEAUTIFUL RESULT CARD
                st.markdown(f"""
                <div class="diagnosis-card" style="border-top: 6px solid {border_color}; background-color: {bg_color};">
                    <h4 style="margin:0; color:#555 !important;">AI Prediction</h4>
                    <h1 style="margin:10px 0; color:#1e293b !important;">{icon} {full_name}</h1>
                    <div style="display:flex; justify-content:center; gap:20px; margin-top:15px;">
                        <span style="background:white; padding:5px 15px; border-radius:15px; border:1px solid {border_color}; color:{border_color}; font-weight:bold;">
                            {status}
                        </span>
                        <span style="background:white; padding:5px 15px; border-radius:15px; border:1px solid #3498db; color:#3498db; font-weight:bold;">
                            {conf*100:.1f}% Confidence
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Detailed Chart
                st.write("")
                st.markdown("#### Differential Probabilities")
                
                # Create a nice clean table
                res_df = pd.DataFrame({
                    "Condition": [class_details[c][0] for c in classes],
                    "Risk Level": [class_details[c][1] for c in classes],
                    "Probability": [f"{p*100:.2f}%" for p in preds[0]]
                }).sort_values(by="Probability", ascending=False)
                
                # Using standard table for maximum compatibility
                st.table(res_df)

    else:
        # Empty State
        st.info("👈 Waiting for image upload...")
        st.markdown("""
        **Supported Classifications:**
        * **Melanoma** (High Risk)
        * **Carcinoma** (Basal Cell)
        * **Nevi** (Benign Moles)
        * **Keratosis** (Benign/Pre-cancerous)
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 12px;">
    <strong>Medical Disclaimer:</strong> This tool is for educational purposes only. 
    AI predictions should never replace professional medical advice. 
    Always consult a dermatologist for diagnosis.
</div>
""", unsafe_allow_html=True)
