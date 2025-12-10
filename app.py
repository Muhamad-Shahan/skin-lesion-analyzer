import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Skin Lesion Analyzer", page_icon="🩺", layout="wide")

# Columns expected by the model (Order matters! Must match training)
# Based on standard HAM10000 encoding
META_COLUMNS = [
    'age_norm', 
    'sex_female', 'sex_male', 'sex_unknown',
    'localization_abdomen', 'localization_acral', 'localization_back', 
    'localization_chest', 'localization_ear', 'localization_face', 
    'localization_foot', 'localization_genital', 'localization_hand', 
    'localization_lower extremity', 'localization_neck', 'localization_scalp', 
    'localization_trunk', 'localization_upper extremity', 'localization_unknown'
]

CLASS_NAMES = ['Actinic Keratoses (akiec)', 'Basal Cell Carcinoma (bcc)', 
               'Benign Keratosis (bkl)', 'Dermatofibroma (df)', 
               'Melanoma (mel)', 'Melanocytic Nevi (nv)', 'Vascular Lesions (vasc)']

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    # Load the model we trained in Colab
    return tf.keras.models.load_model('best_skin_model.keras')

try:
    model = load_model()
    st.success("Model loaded successfully! System Ready.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- 3. UI LAYOUT ---
st.title("🩺 Multimodal Skin Lesion Classifier")
st.markdown("Upload a dermatoscopic image and provide patient details for an AI-assisted diagnosis.")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Patient Data")
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    sex = st.selectbox("Sex", ["Male", "Female", "Unknown"])
    loc = st.selectbox("Localization (Body Part)", [
        "Back", "Lower Extremity", "Trunk", "Upper Extremity", "Abdomen", 
        "Face", "Chest", "Foot", "Neck", "Scalp", "Hand", "Ear", "Genital", "Acral"
    ])

    st.header("2. Image Upload")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# --- 4. PREDICTION LOGIC ---
if uploaded_file is not None:
    # A. Display Image
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # B. Preprocess Image (Visual Input)
    # Resize to 224x224
    img_array = image.resize((224, 224))
    img_array = np.array(img_array)
    
    # Ensure 3 channels (RGB)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
        
    # Standard ResNet50 Preprocessing (Crucial!)
    # This converts RGB to BGR and centers the pixels (Zero-centering)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)

    # C. Preprocess Metadata (Tabular Input)
    # Create a DataFrame with zeros for all columns
    meta_df = pd.DataFrame(0, index=[0], columns=META_COLUMNS)
    
    # Fill Age (Normalized)
    meta_df['age_norm'] = age / 100.0
    
    # Fill Sex (One-Hot)
    sex_col = f"sex_{sex.lower()}"
    if sex_col in meta_df.columns:
        meta_df[sex_col] = 1
        
    # Fill Localization (One-Hot)
    # Note: Dataset uses lowercase 'lower extremity', UI uses 'Lower Extremity'
    loc_clean = loc.lower()
    loc_col = f"localization_{loc_clean}"
    if loc_col in meta_df.columns:
        meta_df[loc_col] = 1
        
    # Convert to numpy array
    meta_array = meta_df.values.astype('float32')

    # D. Prediction
    if st.button("Analyze Lesion"):
        with st.spinner("AI is analyzing features..."):
            # Pass dictionary matching input layer names
            predictions = model.predict({'image_input': img_preprocessed, 'meta_input': meta_array})
            score = tf.nn.softmax(predictions[0])
            
            # Get top prediction
            class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            
        # --- 5. RESULTS DISPLAY ---
        with col2:
            st.header("3. Diagnosis Results")
            
            # Color code the result (Red for Cancer, Green for Benign)
            # Dangerous: mel, bcc, akiec
            # Benign: nv, bkl, df, vasc
            danger_classes = [0, 1, 4] # indices for akiec, bcc, mel
            
            if class_idx in danger_classes:
                st.error(f"### Prediction: {CLASS_NAMES[class_idx]}")
                st.write("⚠️ **High Risk Alert:** This lesion shows features consistent with malignancy.")
            else:
                st.success(f"### Prediction: {CLASS_NAMES[class_idx]}")
                st.write("✅ **Low Risk:** This lesion appears benign.")
            
            st.metric("Confidence Score", f"{confidence:.2f}%")
            
            st.subheader("Probability Distribution")
            chart_data = pd.DataFrame(
                predictions[0],
                index=CLASS_NAMES,
                columns=['Probability']
            )
            st.bar_chart(chart_data)
            
            st.info("Note: This is an AI-assisted tool and not a substitute for professional medical advice.")