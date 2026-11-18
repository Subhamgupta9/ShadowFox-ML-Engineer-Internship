# ShadowFox — Machine Learning Engineer Internship

[![Python Version](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15%20CPU-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-5.23-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 📖 Internship Overview

This repository hosts all project deliverables, task implementations, research notebooks, and interactive applications completed during the **Machine Learning Engineer Internship** at **ShadowFox**.

The internship curriculum spans foundational, intermediate, and advanced applied machine learning:
1. **Week 1 (Foundational):** Deep Learning & Computer Vision (Transfer Learning with MobileNetV2, CNN fine-tuning, Streamlit deployment).
2. **Week 2 (Intermediate):** Tabular Machine Learning & Regression (Exploratory Data Analysis, Multi-Model Benchmarking, Extra Trees Regressor, Streamlit automotive dashboard).
3. **Week 3 (Advanced):** Foundation Language Models & Natural Language Processing (FLAN-T5-small Seq2Seq deployment, multi-task benchmarking, decoding strategy analysis, interactive research laboratory).

---

## 🗂️ Repository Structure

```text
ShadowFox-ML-Engineer-Internship/
│
├── .gitignore                                # Global Python, cache & ML exclusion rules
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
│       ├── tests/                            # Automated Pytest suite (8 unit tests)
│       ├── requirements.txt                  # Task 2 dependencies
│       └── README.md                         # Detailed Task 2 documentation & benchmark analysis
│
└── Week-3/
    └── Task-3-Language-Model-Deployment/     # NEXUS NLP — Language Model Deployment & Evaluation Lab
        ├── app/                              # Streamlit 4-tab interactive research laboratory
        ├── notebooks/                        # 19-section executed research notebook (nexus_nlp_lab.ipynb)
        ├── reports/                          # Benchmark results CSV, JSON summaries & figures
        ├── src/                              # Modular inference engine (FLAN-T5), metrics & runner
        ├── tests/                            # Automated Pytest suite (23 unit tests)
        ├── requirements.txt                  # Task 3 dependencies (PyTorch CPU, Transformers, etc.)
        └── README.md                         # Comprehensive Task 3 research report & documentation
```

---

## 📋 Task Status & Curriculum Roadmap

| Week | Task | Project Name | Domain / Stack | Status | Metric / Highlights |
| :---: | :---: | :--- | :--- | :---: | :--- |
| **Week 1** | Task 1 | [Image Tagging with TensorFlow](Week-1/Task-1-Image-Tagging/) | Computer Vision · MobileNetV2 · Streamlit | ✅ **Completed** | **98.52% Test Acc**, **98.40% Val Acc**, Hyperparameter Tuning verified |
| **Week 2** | Task 2 | [CARSPHERE AI — Car Resale Valuation](Week-2/Task-2-Car-Selling-Price-Prediction/) | Tabular Regression · Extra Trees · Streamlit | ✅ **Completed** | **Test R² = 0.8119**, **Test MAE = ₹1.004L**, **5-Fold CV R² = 0.8973**, 8 Pytest unit tests |
| **Week 3** | Task 3 | [NEXUS NLP — Language Model Lab](Week-3/Task-3-Language-Model-Deployment/) | NLP · FLAN-T5 (77M) · PyTorch CPU · Streamlit | ✅ **Completed** | **80.0% QA F1**, **83.33% Classif. Acc**, ROUGE/SacreBLEU evaluation, 23 Pytest unit tests, 4-Tab Web App |

---

## 🛠️ Technologies & Ecosystem

- **Programming Languages:** Python 3.11+
- **Deep Learning Frameworks:** PyTorch 2.15+ (CPU), TensorFlow 2.21.0, Keras 3.15.1
- **NLP & Transformers Ecosystem:** Hugging Face Transformers 5.23+, Tokenizers, SentencePiece, Accelerate, Evaluate, ROUGE-Score, SacreBLEU
- **Machine Learning & Tabular Models:** Scikit-learn 1.5+, Extra Trees Regressor, Gradient Boosting, Random Forest, Ridge Regression
- **Data Engineering & Scientific Computing:** Pandas 2.2+, NumPy 1.26+, SciPy 1.14+
- **Visualization & Plotting:** Matplotlib 3.9+, Seaborn 0.13+, Plotly 5.20+
- **Web Applications & Dashboards:** Streamlit 1.35+
- **Testing, Quality Assurance & Automation:** Pytest 8.3+, Playwright
- **Hardware Target:** CPU Optimized (Intel x86_64, zero GPU dependency)
- **Version Control:** Git, GitHub, Git LFS hygiene

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
*Open `http://localhost:8501` to test the image classifier. See [Week 1 Task 1 README](Week-1/Task-1-Image-Tagging/README.md).*

### 3. Week 2: CARSPHERE AI — Car Selling Price Prediction
```bash
cd Week-2/Task-2-Car-Selling-Price-Prediction
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
pytest tests/ -v
streamlit run app/app.py --server.port 8502
```
*Open `http://localhost:8502` to test the vehicle valuation dashboard. See [Week 2 Task 2 README](Week-2/Task-2-Car-Selling-Price-Prediction/README.md).*

### 4. Week 3: NEXUS NLP — Language Model Deployment & Intelligence Evaluation Lab
```bash
cd Week-3/Task-3-Language-Model-Deployment
python -m venv venv
.\venv\Scripts\activate
pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest tests/ -v
streamlit run app/app.py --server.port 8503
```
*Open `http://localhost:8503` to launch the interactive research laboratory. See [Week 3 Task 3 README](Week-3/Task-3-Language-Model-Deployment/README.md).*

---

## 👤 Author & Acknowledgements

- **Intern:** [Subham Gupta](https://github.com/Subhamgupta9)
- **Role:** Machine Learning Engineer Intern
- **Organization:** [ShadowFox](https://shadowfox.in/)
- **GitHub Profile:** [@Subhamgupta9](https://github.com/Subhamgupta9)
