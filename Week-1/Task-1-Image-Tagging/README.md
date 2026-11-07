# Image Tagging with TensorFlow

**Organization:** ShadowFox  
**Internship Role:** Machine Learning Engineer  
**Domain:** Computer Vision & Deep Learning  
**Task:** Week 1 — Task 1: Image Tagging / Classification  

---

## 📌 Project Overview

This project delivers an end-to-end, high-accuracy deep learning image classification system developed during the **ShadowFox Machine Learning Engineer Internship**. 

The system automates the process of image tagging and binary classification between **Cats** and **Dogs** using a transfer learning pipeline built on top of **MobileNetV2**. An interactive, modern web application powered by **Streamlit** provides an intuitive drag-and-drop interface where users can upload any standard image format (`.jpg`, `.jpeg`, `.png`), analyze it in real time, and view clear predictions alongside calibrated confidence scores.

### Key Highlights
- **Production-Ready Inference Pipeline:** Lightweight inference engine optimized for low-latency image tagging.
- **High-Accuracy Classifier:** Achieved **98.52% Test Accuracy** on held-out evaluation sets.
- **Robust Model Deserialization:** Seamlessly bridges TensorFlow 2.x and Keras 3.x with a custom object registry for MobileNetV2 preprocessing.
- **Responsive UI:** Streamlit interface styled with custom CSS cards, accessible status tags, and interactive confidence meters.

---

## 🖼️ Application Screenshots & Prediction Examples

### 1. Interactive Classification Dashboard
The user interface allows quick image uploads with immediate visual feedback and confidence tracking:

![Prediction Result](screenshots/prediction_result.png)

### 2. Streamlit Clean Upload State
A distraction-free, modern interface ready for image ingestion:

![Application Overview](screenshots/app_ui_overview.png)

### 3. Example Prediction Output
| Input Image | Actual Class | Predicted Tag | Model Confidence | Status |
| :---: | :---: | :---: | :---: | :---: |
| `sample_dog.jpg` | Dog | **Dog** | **99.91%** | ✅ Correct |
| `sample_cat.jpg` | Cat | **Cat** | **98.84%** | ✅ Correct |

---

## 🛠️ Technologies & Libraries Used

| Technology / Library | Version | Purpose |
| :--- | :--- | :--- |
| **Python** | 3.11+ | Core programming environment |
| **TensorFlow** | 2.21.0 | Deep learning framework & tensor computation |
| **Keras** | 3.15.1 | High-level neural network API and model serialization |
| **Streamlit** | 1.67.0 | Web application framework for interactive UI |
| **Pillow (PIL)** | 13.0.0 | Image ingestion, decoding, and preprocessing |
| **NumPy** | 2.4.6 | Numerical matrix transformations & array expansion |

---

## 🧠 Model Architecture & Design

The model utilizes transfer learning using Google's **MobileNetV2** backbone architecture pre-trained on the ImageNet dataset. MobileNetV2 uses depthwise separable convolutions and inverted residual structures with linear bottlenecks, making it exceptionally fast, memory-efficient, and suitable for edge deployment.

### Network Topology
```text
Input Image (160 × 160 × 3, RGB)
       │
       ▼
[Data Augmentation / Sequential Preprocessing]
       │
       ▼
[Lambda Layer: MobileNetV2 Preprocessing] (Scales pixels to [-1, 1])
       │
       ▼
[MobileNetV2 Base] (1.00_160, Feature Extractor: 1280 feature maps)
       │
       ▼
[GlobalAveragePooling2D] (Reduces 5 × 5 × 1280 to 1280 feature vector)
       │
       ▼
[Dropout Layer] (Regularization rate: 0.2 to prevent overfitting)
       │
       ▼
[Dense Output Layer] (1 unit, linear logit output)
       │
       ▼
[Sigmoid Activation (Inference)] -> P(Dog) >= 0.5 ? Dog : Cat
```

### Parameter Breakdown
- **Total Parameters:** `2,259,265` (~8.62 MB weights)
- **Trainable Parameters:** `1,281` (Classification Head)
- **Non-trainable Parameters:** `2,257,984` (Pre-trained MobileNetV2 Feature Extractor)
- **Input Dimensions:** `(None, 160, 160, 3)`
- **Output:** Binary logit mapped via $\sigma(z) = \frac{1}{1 + e^{-z}}$ to calculate class confidence.

---

## 📊 Dataset, Training & Evaluation

- **Dataset:** Kaggle / Microsoft Cats vs. Dogs dataset curated into balanced training, validation, and test subsets.
- **Preprocessing & Augmentation:** Random horizontal flipping and slight random rotation applied during training to improve domain generalization. Inputs normalized to `[-1, 1]` matching MobileNetV2 specifications.
- **Optimization Strategy:** Binary Cross-Entropy loss optimized via Adam optimizer.
- **Evaluation Metric:**
  - **Test Accuracy:** **98.52%**
  - **Validation Accuracy:** **98.40%**
  - **Inference Latency:** < 50ms per image on standard CPU.

---

## ⚙️ Model Deserialization & Keras 3 Compatibility

When loading the trained `.keras` artifact across environments running Keras 3.x / TensorFlow 2.x, the embedded `Lambda` preprocessing layer requires explicit mapping of the preprocessing function.

In `app.py`, this is handled robustly via:

```python
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

@st.cache_resource
def load_model():
    """Load the trusted inference model and its MobileNetV2 preprocessing layer."""
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": preprocess_input},
        compile=False,
    )
```

This guarantees seamless model loading on any machine without deserialization warnings or errors.

---

## 💾 Model File Size & GitHub Tracking

- **Model File:** `shadowfox_cats_dogs_mobilenetv2.keras`
- **File Size:** **9.20 MB** (9,646,338 bytes)
- **GitHub Policy Check:**
  - GitHub's soft warning threshold is **50 MB**.
  - GitHub's hard file rejection limit is **100 MB**.
  - At **9.2 MB**, this trained model is well under all limits and can be tracked natively in standard Git without requiring Git LFS.

> [!NOTE]
> If future tasks require models exceeding 100 MB (e.g., full uncompressed ViT, ResNet-152, or LLMs), **Git Large File Storage (Git LFS)** should be used:
> ```bash
> git lfs install
> git lfs track "*.keras" "*.h5" "*.pt"
> git add .gitattributes
> ```

---

## 🚀 Installation & Setup Guide

### Prerequisites
- Python 3.10 or 3.11 installed.
- Git and GitHub Desktop installed.

### 1. Clone or Open the Repository
```bash
git clone https://github.com/<your-username>/ShadowFox-ML-Engineer-Internship.git
cd ShadowFox-ML-Engineer-Internship/Week-1/Task-1-Image-Tagging
```

### 2. Create and Activate a Virtual Environment
- **On Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **On macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🖥️ Launching the Application

Run the Streamlit web application from within the `Week-1/Task-1-Image-Tagging` folder:

```bash
streamlit run app.py
```

Once executed, your browser will automatically open:
```text
Local URL: http://localhost:8501
```

1. Drag and drop any image of a cat or dog into the upload zone.
2. Click **"Analyze image"**.
3. View the classified tag and confidence score instantly!
