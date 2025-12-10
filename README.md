# 🩺 DermaVision Pro: Multimodal Skin Lesion Analyzer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://skin-lesion-analyzer-macrkhfljjpy7ahxeqy7fv.streamlit.app/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![Model](https://img.shields.io/badge/Model-ResNet50_Multimodal-blue.svg)](https://arxiv.org/abs/1512.03385)
[![Dataset](https://img.shields.io/badge/Data-HAM10000-green.svg)](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

## 📄 Abstract
Melanoma is the deadliest form of skin cancer, but survival rates exceed 95% if detected early. Traditional AI tools often rely solely on image data, missing crucial patient context.

**DermaVision Pro** is a **Multimodal Deep Learning System** that mimics the clinical diagnostic process. It fuses **Dermatoscopic Imaging (CNN)** with **Patient Metadata (Age, Sex, Anatomical Site)** to classify skin lesions into 7 diagnostic categories with high precision. The system is deployed via a high-contrast, accessibility-focused Streamlit interface designed for clinical usability.

> **[🔴 Launch Live Diagnostic Tool](https://skin-lesion-analyzer-macrkhfljjpy7ahxeqy7fv.streamlit.app/)**

## 🧠 Diagnostic Capabilities (The HAM10000 Index)
The model is trained to detect the following conditions based on the **HAM10000** ("Human Against Machine with 10000 training images") dataset:

| Class | Diagnosis | Clinical Significance |
|:-----:|-----------|-----------------------|
| **mel** | **Melanoma** | 🚨 **High Risk:** Malignant skin cancer. |
| **bcc** | **Basal Cell Carcinoma** | 🚨 **High Risk:** Common malignant growth. |
| **akiec** | **Actinic Keratoses** | ⚠️ **Risk:** Pre-cancerous / intraepithelial carcinoma. |
| **nv** | Melanocytic Nevi | ✅ Benign: Common mole. |
| **bkl** | Benign Keratosis | ✅ Benign: Seborrheic keratosis/Lentigo. |
| **df** | Dermatofibroma | ✅ Benign: Skin nodule. |
| **vasc** | Vascular Lesions | ✅ Benign: Cherry angiomas, etc. |

## 🛠️ Technical Architecture
This project implements a **Dual-Input Neural Network**:

1.  **Visual Stream (ResNet50):**
    * Uses **Transfer Learning** with a ResNet50 backbone (ImageNet weights) to extract spatial features from 224x224 skin images.
    * *Preprocessing:* `tf.keras.applications.resnet50.preprocess_input` (Zero-centering).

2.  **Metadata Stream (Dense Network):**
    * Processes structured clinical data (Age, Sex, Localization).
    * *Encoding:* One-Hot Encoding matches the exact feature space of the training set.

3.  **Fusion Layer:**
    * Concatenates the 2048-dimensional image vector with the clinical metadata vector.
    * Passes through dense layers for final Softmax classification.

## 📦 Installation & Usage

**Prerequisites:** Python 3.9+, TensorFlow 2.10+

```bash
# 1. Clone the repository
git clone [https://github.com/Muhammad-Shahan/DermaVision-Pro.git](https://github.com/Muhammad-Shahan/DermaVision-Pro.git)
cd DermaVision-Pro

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Application
streamlit run app.py
