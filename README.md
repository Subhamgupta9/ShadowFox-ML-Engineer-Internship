# ShadowFox — Machine Learning Engineer Internship

[![Python Version](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.15-D00000?style=flat&logo=keras&logoColor=white)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 📖 Internship Overview

This repository hosts all project deliverables, task implementations, and documentation completed during the **Machine Learning Engineer Internship** at **ShadowFox**. 

The internship emphasizes hands-on machine learning engineering, spanning:
- Deep Learning & Computer Vision (Transfer Learning, Convolutional Neural Networks).
- Model Evaluation, Optimization, and Deserialization across production environments.
- Interactive AI application deployment using modern frameworks like Streamlit.
- Industry-standard Git and GitHub engineering practices.

---

## 🗂️ Repository Structure

```text
ShadowFox-ML-Engineer-Internship/
│
├── .gitignore                                # Global Python & ML exclusion rules
├── README.md                                 # Main internship portfolio overview
│
├── Week-1/
│   └── Task-1-Image-Tagging/                 # Image Tagging with TensorFlow
│       ├── app.py                            # Streamlit web application & inference UI
│       ├── requirements.txt                  # Python dependencies
│       ├── shadowfox_cats_dogs_mobilenetv2.keras # Production MobileNetV2 classifier (9.2 MB)
│       ├── shadowfox_cats_dogs_mobilenetv2_finetuned_best.keras # Preserved fine-tuned checkpoint (9.2 MB)
│       ├── experiment_results.csv            # Hyperparameter tuning benchmark metrics
│       ├── ShadowFox_Week1_Hyperparameter_Tuning.ipynb # Google Colab experiment notebook
│       ├── README.md                         # Detailed Task 1 documentation & metrics
│       └── screenshots/                      # UI and tuning evaluation screenshots
│           ├── app_ui_overview.png
│           ├── prediction_result.png
│           ├── sample_dog.jpg
│           └── tuning_comparison_chart.png
│
├── Week-2/
│   └── .gitkeep                              # Reserved for Week 2 deliverables
│
└── Week-3/
    └── .gitkeep                              # Reserved for Week 3 deliverables
```

---

## 📋 Task Status & Curriculum Roadmap

| Week | Task | Project Name | Domain / Stack | Status | Metric / Highlights |
| :---: | :---: | :--- | :--- | :---: | :--- |
| **Week 1** | Task 1 | [Image Tagging with TensorFlow](Week-1/Task-1-Image-Tagging/) | Computer Vision · MobileNetV2 · Streamlit | ✅ **Completed** | **98.52% Test Acc**, **98.40% Val Acc**, Hyperparameter Tuning verified |
| **Week 2** | Task 2 | *To be announced / upcoming* | Machine Learning / NLP / Tabular | ⏳ In Progress | Task milestone in development |
| **Week 3** | Task 3 | *To be announced / upcoming* | Advanced ML / Deployment | ⏳ Upcoming | Final internship deliverable |

---

## 🛠️ Technologies & Ecosystem

- **Programming Language:** Python 3.11+
- **Deep Learning Frameworks:** TensorFlow 2.21.0, Keras 3.15.1
- **Computer Vision & Image Processing:** Pillow (PIL) 13.0.0, MobileNetV2
- **Web App & Deployment:** Streamlit 1.67.0
- **Scientific Computing:** NumPy 2.4.6
- **Version Control & Management:** Git, GitHub, GitHub Desktop

---

## 🚀 Quick Start — Running Completed Projects

To explore or run any completed task locally:

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/ShadowFox-ML-Engineer-Internship.git
cd ShadowFox-ML-Engineer-Internship
```

### 2. Navigate to the Task Directory
For Week 1 (Image Tagging):
```bash
cd Week-1/Task-1-Image-Tagging
```

### 3. Create a Virtual Environment and Install Dependencies
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser to test the interactive image classifier.

For full architectural details, training benchmarks, and prediction samples, see the [Week 1 Task 1 README](Week-1/Task-1-Image-Tagging/README.md).

---

## 👤 Author & Acknowledgements

- **Intern:** Divyanshi Gupta
- **Role:** Machine Learning Engineer Intern
- **Organization:** [ShadowFox](https://shadowfox.in/)
