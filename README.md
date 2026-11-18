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
│   └── Task-2-Car-Selling-Price-Prediction/  # CARSPHERE AI — Vehicle Valuation Engine
│       ├── app/                              # Streamlit web application & styling utils
│       ├── dataset/                          # Secondary market vehicle dataset (car.csv)
│       ├── models/                           # Serialized Extra Trees pipeline (car_price_model.joblib)
│       ├── notebooks/                        # EDA, model benchmarking & tuning notebook
│       ├── reports/                          # Benchmark CSV & 10 publication figures
│       ├── screenshots/                      # UI validation & governance captures
│       ├── src/                              # Preprocessing, evaluation & prediction engine
│       ├── tests/                            # Automated Pytest suite
│       ├── requirements.txt                  # Task 2 dependencies
│       └── README.md                         # Detailed Task 2 documentation & benchmark analysis
│
└── Week-3/
    └── .gitkeep                              # Reserved for Week 3 deliverables
```

---

## 📋 Task Status & Curriculum Roadmap

| Week | Task | Project Name | Domain / Stack | Status | Metric / Highlights |
| :---: | :---: | :--- | :--- | :---: | :--- |
| **Week 1** | Task 1 | [Image Tagging with TensorFlow](Week-1/Task-1-Image-Tagging/) | Computer Vision · MobileNetV2 · Streamlit | ✅ **Completed** | **98.52% Test Acc**, **98.40% Val Acc**, Hyperparameter Tuning verified |
| **Week 2** | Task 2 | [CARSPHERE AI — Car Resale Valuation](Week-2/Task-2-Car-Selling-Price-Prediction/) | Tabular Regression · Extra Trees · Streamlit | ✅ **Completed** | **Test R² = 0.8119**, **Test MAE = ₹1.004L**, **5-Fold CV R² = 0.8973**, 8 Pytest unit tests |
| **Week 3** | Task 3 | *To be announced / upcoming* | Advanced ML / Deployment | ⏳ Upcoming | Final internship deliverable |

---

## 🛠️ Technologies & Ecosystem

- **Programming Language:** Python 3.11+
- **Deep Learning Frameworks:** TensorFlow 2.21.0, Keras 3.15.1
- **Machine Learning & Regression:** Scikit-learn 1.3+, Extra Trees Regressor, Gradient Boosting, Ridge Regression
- **Data Engineering & Analysis:** Pandas 2.0+, NumPy 1.24+
- **Visualization & Reporting:** Matplotlib 3.7+, Seaborn 0.12+, Pillow (PIL) 13.0+
- **Web App & Deployment:** Streamlit 1.30+
- **Model Serialization & Quality Assurance:** Joblib 1.3+, Pytest 7.0+
- **Browser Automation & Inspection:** Playwright
- **Version Control & Management:** Git, GitHub, GitHub Desktop

---

## 🚀 Quick Start — Running Completed Projects

To explore or run any completed task locally:

### 1. Clone the Repository
```bash
git clone https://github.com/Subhamgupta9/ShadowFox-ML-Engineer-Internship.git
cd ShadowFox-ML-Engineer-Internship
```

### 2. Week 1: Image Tagging with TensorFlow
```bash
cd Week-1/Task-1-Image-Tagging
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Open `http://localhost:8501` to test the interactive image classifier. See [Week 1 Task 1 README](Week-1/Task-1-Image-Tagging/README.md).

### 3. Week 2: CARSPHERE AI — Car Selling Price Prediction
```bash
cd Week-2/Task-2-Car-Selling-Price-Prediction
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
pytest tests/ -v
streamlit run app/app.py --server.port 8502
```
Open `http://localhost:8502` to test the vehicle valuation dashboard. See [Week 2 Task 2 README](Week-2/Task-2-Car-Selling-Price-Prediction/README.md).

---

## 👤 Author & Acknowledgements

- **Intern:** [Subham Gupta](https://github.com/Subhamgupta9)
- **Role:** Machine Learning Engineer Intern
- **Organization:** [ShadowFox](https://shadowfox.in/)
- **GitHub:** [@Subhamgupta9](https://github.com/Subhamgupta9)
