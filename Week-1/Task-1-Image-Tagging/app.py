
import numpy as np
from pathlib import Path
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


MODEL_PATH = Path(__file__).with_name("shadowfox_cats_dogs_mobilenetv2.keras")

st.set_page_config(
    page_title="Image Tagging with TensorFlow",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --page: #f7f8fc;
                --surface: #ffffff;
                --surface-muted: #f2f5fb;
                --ink: #17223b;
                --muted: #64748b;
                --line: #e4e9f2;
                --accent: #4f46e5;
                --accent-dark: #3730a3;
                --accent-soft: #eef2ff;
                --success: #13795b;
                --shadow: 0 12px 32px rgba(26, 42, 71, 0.08);
            }
            .stApp { background: var(--page); color: var(--ink); }
            [data-testid="stHeader"] { background: rgba(247, 248, 252, 0.92); }
            .block-container { max-width: 1080px; padding-top: 2.75rem; padding-bottom: 2rem; }
            .project-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 1.25rem; padding: 0 0.1rem 2rem; }
            .eyebrow, .model-badge { display: inline-flex; align-items: center; gap: 0.45rem; width: fit-content; border-radius: 999px; font-size: 0.73rem; font-weight: 750; letter-spacing: 0.08em; text-transform: uppercase; }
            .eyebrow { color: var(--accent-dark); margin: 0 0 0.7rem; }
            .eyebrow::before { content: ""; width: 0.45rem; height: 0.45rem; border-radius: 50%; background: var(--accent); box-shadow: 0 0 0 4px var(--accent-soft); }
            .project-header h1 { color: var(--ink); font-size: clamp(2rem, 5vw, 3.35rem); font-weight: 760; letter-spacing: -0.055em; line-height: 1.04; margin: 0; }
            .project-header p { color: var(--muted); font-size: 1rem; margin: 0.85rem 0 0; }
            .model-badge { color: var(--accent-dark); background: var(--accent-soft); border: 1px solid #dfe4ff; padding: 0.62rem 0.82rem; white-space: nowrap; }
            .model-badge svg { width: 0.95rem; height: 0.95rem; fill: currentColor; }
            .section-card, .result-card, .model-information, [data-testid="stVerticalBlockBorderWrapper"] { background: var(--surface); border: 1px solid var(--line); border-radius: 20px; box-shadow: var(--shadow); }
            [data-testid="stVerticalBlockBorderWrapper"] { padding: clamp(1.25rem, 3vw, 2rem); }
            .section-label { color: var(--accent-dark); font-size: 0.72rem; font-weight: 750; letter-spacing: 0.09em; text-transform: uppercase; margin: 0 0 0.5rem; }
            .section-card h2, .model-information h2 { color: var(--ink); font-size: 1.55rem; letter-spacing: -0.03em; margin: 0; }
            .section-description { color: var(--muted); line-height: 1.65; margin: 0.65rem 0 1.35rem; max-width: 38rem; }
            [data-testid="stFileUploaderDropzone"] { min-height: 185px; background: #fbfcff; border: 1.5px dashed #bdc8e8; border-radius: 16px; padding: 1.35rem; transition: border-color 180ms ease, box-shadow 180ms ease, transform 180ms ease, background-color 180ms ease; }
            [data-testid="stFileUploaderDropzone"]:hover, [data-testid="stFileUploaderDropzone"]:focus-within { background: #f8f9ff; border-color: var(--accent); box-shadow: 0 8px 20px rgba(79, 70, 229, 0.12); transform: translateY(-1px); }
            [data-testid="stFileUploaderDropzone"] > div { color: var(--muted); }
            [data-testid="stFileUploaderDropzone"] button, .stButton > button { border-radius: 10px; font-weight: 700; transition: background-color 180ms ease, box-shadow 180ms ease, transform 180ms ease; }
            [data-testid="stFileUploaderDropzone"] button { color: var(--accent-dark); background: var(--accent-soft); border: 1px solid #dbe2ff; }
            .stButton > button[kind="primary"] { min-height: 2.8rem; color: #ffffff; background: var(--accent); border: 1px solid var(--accent); }
            .stButton > button[kind="primary"]:hover { background: var(--accent-dark); border-color: var(--accent-dark); box-shadow: 0 8px 16px rgba(79, 70, 229, 0.23); transform: translateY(-1px); }
            .upload-hint { color: var(--muted); font-size: 0.83rem; margin: 0.95rem 0 0; }
            .result-area { margin-top: 1.35rem; }
            .image-kicker { color: var(--muted); font-size: 0.78rem; font-weight: 700; margin: 0 0 0.6rem; }
            [data-testid="stImage"] img { border: 1px solid var(--line); border-radius: 16px; box-shadow: 0 10px 24px rgba(26, 42, 71, 0.08); }
            .result-card { min-height: 100%; padding: 1.35rem; }
            .result-overline { color: var(--success); font-size: 0.73rem; font-weight: 750; letter-spacing: 0.08em; margin: 0 0 0.6rem; text-transform: uppercase; }
            .prediction-name { color: var(--ink); font-size: clamp(2rem, 4.2vw, 2.8rem); font-weight: 760; letter-spacing: -0.05em; line-height: 1; margin: 0; }
            .confidence-row { align-items: baseline; display: flex; gap: 0.4rem; margin: 1.75rem 0 0.65rem; }
            .confidence-value { color: var(--accent-dark); font-size: 1.85rem; font-weight: 760; letter-spacing: -0.04em; }
            .confidence-label { color: var(--muted); font-size: 0.88rem; font-weight: 600; }
            .confidence-track { width: 100%; height: 0.62rem; overflow: hidden; background: #e9edf7; border-radius: 999px; }
            .confidence-fill { display: block; height: 100%; border-radius: inherit; background: var(--accent); transition: width 300ms ease-out; }
            .result-note { color: var(--muted); font-size: 0.84rem; line-height: 1.5; margin: 1rem 0 0; }
            .model-information { margin-top: 2rem; padding: clamp(1.25rem, 3vw, 1.75rem); }
            .model-information h2 { font-size: 1.2rem; }
            .model-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.75rem; margin-top: 1.2rem; }
            .model-item { background: var(--surface-muted); border-radius: 12px; padding: 0.9rem; }
            .model-item span { color: var(--muted); display: block; font-size: 0.71rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; }
            .model-item strong { color: var(--ink); display: block; font-size: 0.98rem; letter-spacing: -0.015em; margin-top: 0.28rem; }
            .footer { color: var(--muted); font-size: 0.8rem; margin: 2rem 0 0; text-align: center; }
            @media (max-width: 700px) { .block-container { padding: 1.5rem 1rem 1.75rem; } .project-header { align-items: flex-start; flex-direction: column; gap: 1rem; padding-bottom: 1.5rem; } .model-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
            @media (prefers-reduced-motion: reduce) { *, *::before, *::after { scroll-behavior: auto !important; transition-duration: 0.01ms !important; } }
        </style>
        """,
        unsafe_allow_html=True,
    )

@st.cache_resource
def load_model():
    """Load the trusted inference model and its MobileNetV2 preprocessing layer."""
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": preprocess_input},
        compile=False,
    )

def classify_image(image: Image.Image) -> tuple[str, float]:
    """Return the predicted label and its confidence for one uploaded image."""
    resized_image = image.resize((160, 160))
    image_array = np.expand_dims(
        np.array(resized_image, dtype=np.float32), axis=0
    )
    prediction = model.predict(image_array, verbose=0)
    dog_probability = float(tf.sigmoid(prediction[0][0]))

    if dog_probability >= 0.5:
        return "Dog", dog_probability * 100
    return "Cat", (1 - dog_probability) * 100


inject_styles()
model = load_model()

st.markdown(
    """
    <header class="project-header">
        <div>
            <div class="eyebrow">Computer Vision Project</div>
            <h1>Image Tagging with TensorFlow</h1>
            <p>ShadowFox | Machine Learning Engineer Internship</p>
        </div>
        <div class="model-badge" aria-label="TensorFlow machine learning model">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2 2.9 7.25v10.5L12 23l9.1-5.25V7.25L12 2Zm0 3.1 6.4 3.7-2.26 1.3L12 7.72 7.86 10.1 5.6 8.8 12 5.1Zm-7 6.26 2.2 1.27v4.74L5 16.1v-4.74Zm4.4 2.54 1.5.87v5.3l-1.5-.87v-5.3Zm3.1.87 1.5-.87v5.3l-1.5.87v-5.3Zm3.1-2.14 2.2-1.27v4.74l-2.2 1.27v-4.74Z"/></svg>
            TensorFlow · ML
        </div>
    </header>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown('<p class="section-label">Image Classification</p>', unsafe_allow_html=True)
    st.markdown('<h2>Upload an Image</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-description">Upload a clear image and let the trained MobileNetV2 model identify whether it contains a cat or a dog.</p>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Drop an image here or browse your files",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    st.markdown(
        '<p class="upload-hint">Supported formats: JPG, JPEG, PNG · Best results with a clear, single-subject image.</p>',
        unsafe_allow_html=True,
    )

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except UnidentifiedImageError:
        st.error("This file could not be read as an image. Please upload a JPG, JPEG, or PNG file.")
    else:
        image_column, prediction_column = st.columns((1.05, 0.95), gap="large")

        with image_column:
            st.markdown('<p class="image-kicker">Uploaded image</p>', unsafe_allow_html=True)
            st.image(image, width="stretch")

        with prediction_column:
            if st.button("Analyze image", type="primary", width="stretch"):
                with st.spinner("Analyzing image..."):
                    st.session_state["prediction_result"] = classify_image(image)
                    st.session_state["prediction_file"] = (
                        uploaded_file.name,
                        uploaded_file.size,
                    )

            current_file = (uploaded_file.name, uploaded_file.size)
            if st.session_state.get("prediction_file") == current_file:
                predicted_class, confidence = st.session_state["prediction_result"]
                st.markdown(
                    f"""
                    <section class="result-card">
                        <p class="result-overline">Classification complete</p>
                        <p class="prediction-name">{predicted_class}</p>
                        <div class="confidence-row">
                            <span class="confidence-value">{confidence:.2f}%</span>
                            <span class="confidence-label">confidence</span>
                        </div>
                        <div class="confidence-track" role="progressbar" aria-label="Prediction confidence" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{confidence:.2f}">
                            <span class="confidence-fill" style="width: {confidence:.2f}%"></span>
                        </div>
                        <p class="result-note">Prediction generated by the trained MobileNetV2 cats vs dogs classifier.</p>
                    </section>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("Ready when you are. Select **Analyze image** to view the prediction and confidence score.")

st.markdown(
    """
    <section class="model-information">
        <p class="section-label">Model Information</p>
        <h2>Built for concise, reliable image tagging</h2>
        <div class="model-grid">
            <div class="model-item"><span>Model</span><strong>MobileNetV2</strong></div>
            <div class="model-item"><span>Classification</span><strong>Cats vs Dogs</strong></div>
            <div class="model-item"><span>Test Accuracy</span><strong>98.52%</strong></div>
            <div class="model-item"><span>Framework</span><strong>TensorFlow</strong></div>
        </div>
    </section>
    <p class="footer">Developed as part of the ShadowFox Machine Learning Engineer Internship</p>
    """,
    unsafe_allow_html=True,
)
      
