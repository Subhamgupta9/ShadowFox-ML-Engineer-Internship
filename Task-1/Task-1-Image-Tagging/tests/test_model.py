"""
NEXUS & ShadowFox Machine Learning Engineer Internship
Task 1: Image Tagging with TensorFlow
Automated Unit Tests & Quality Assurance Suite

Tests verify:
1. Framework & library imports
2. Model artifact existence and size integrity (.keras)
3. Image preprocessing pipeline (dimensions, channels, data types)
4. Classification decision boundary & confidence scoring
5. Error handling on corrupted / invalid images
6. End-to-end prediction output contract (label, confidence tuple)
"""

import io
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
try:
    import pytest
except ImportError:
    pytest = None
from PIL import Image, UnidentifiedImageError


TASK1_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = TASK1_DIR / "shadowfox_cats_dogs_mobilenetv2.keras"
FINETUNED_MODEL_FILE = TASK1_DIR / "shadowfox_cats_dogs_mobilenetv2_finetuned_best.keras"


# ==============================================================================
# 1. Dependency & Import Verification
# ==============================================================================

def test_imports():
    """Verify that all core dependencies for Task 1 import cleanly."""
    import numpy as np
    import PIL
    import streamlit as st

    assert np is not None
    assert PIL is not None
    assert st is not None

    try:
        import tensorflow as tf
        assert tf is not None
        assert hasattr(tf.keras, "models")
    except ImportError:
        if pytest is not None:
            pytest.skip("TensorFlow is not installed in the active runner venv; install Task 1 requirements to test TF directly.")
        else:
            pass


# ==============================================================================
# 2. Model Artifact Existence & Integrity
# ==============================================================================

def test_production_model_artifact_exists():
    """Verify the production MobileNetV2 .keras model file exists and is intact."""
    assert MODEL_FILE.exists(), f"Model file not found at: {MODEL_FILE}"
    file_size_mb = MODEL_FILE.stat().st_size / (1024 * 1024)
    # The saved MobileNetV2 classifier is ~9.2 MB
    assert file_size_mb >= 8.0, f"Model file is smaller than expected: {file_size_mb:.2f} MB"


def test_finetuned_model_artifact_exists():
    """Verify the preserved fine-tuned best .keras checkpoint exists and is intact."""
    assert FINETUNED_MODEL_FILE.exists(), f"Finetuned checkpoint not found at: {FINETUNED_MODEL_FILE}"
    file_size_mb = FINETUNED_MODEL_FILE.stat().st_size / (1024 * 1024)
    assert file_size_mb >= 8.0, f"Finetuned checkpoint file is smaller than expected: {file_size_mb:.2f} MB"


# ==============================================================================
# 3. Image Preprocessing Pipeline
# ==============================================================================

def preprocess_image(image: Image.Image, target_size=(160, 160)) -> np.ndarray:
    """Standard preprocessing function mirroring Task 1 inference pipeline."""
    rgb_image = image.convert("RGB")
    resized_image = rgb_image.resize(target_size)
    image_array = np.expand_dims(np.array(resized_image, dtype=np.float32), axis=0)
    return image_array


def test_image_preprocessing_rgb():
    """Verify RGB image preprocessing produces exact expected shape and dtype."""
    sample_img = Image.new("RGB", (320, 240), color=(128, 64, 32))
    processed = preprocess_image(sample_img)
    assert processed.shape == (1, 160, 160, 3)
    assert processed.dtype == np.float32
    assert np.all(processed >= 0.0) and np.all(processed <= 255.0)


def test_image_preprocessing_rgba_conversion():
    """Verify RGBA transparent images are correctly converted to 3-channel RGB."""
    sample_rgba = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    processed = preprocess_image(sample_rgba)
    assert processed.shape == (1, 160, 160, 3)


def test_image_preprocessing_grayscale_conversion():
    """Verify single-channel grayscale images are expanded to 3 RGB channels."""
    sample_gray = Image.new("L", (80, 80), color=200)
    processed = preprocess_image(sample_gray)
    assert processed.shape == (1, 160, 160, 3)


# ==============================================================================
# 4. Classification Decision Boundary & Probability Mapping
# ==============================================================================

def sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    return float(1.0 / (1.0 + np.exp(-x)))


def classify_logit(logit: float) -> tuple[str, float]:
    """Map raw model logit to predicted class and confidence percentage."""
    prob = sigmoid(logit)
    if prob >= 0.5:
        return "Dog", prob * 100.0
    return "Cat", (1.0 - prob) * 100.0


def test_classification_decision_dog():
    """Positive logit represents high dog probability."""
    label, confidence = classify_logit(2.0)
    assert label == "Dog"
    assert confidence > 80.0
    assert confidence <= 100.0


def test_classification_decision_cat():
    """Negative logit represents high cat probability."""
    label, confidence = classify_logit(-2.0)
    assert label == "Cat"
    assert confidence > 80.0
    assert confidence <= 100.0


def test_classification_boundary_threshold():
    """Logit of exactly 0.0 represents the 50.0% boundary."""
    label, confidence = classify_logit(0.0)
    assert label == "Dog"
    assert confidence == 50.0


def test_confidence_range_guarantee():
    """Confidence score must always lie in [50.0, 100.0]."""
    test_logits = [-10.0, -5.0, -1.0, -0.01, 0.0, 0.01, 1.0, 5.0, 10.0]
    for logit in test_logits:
        label, conf = classify_logit(logit)
        assert label in ["Cat", "Dog"]
        assert 50.0 <= conf <= 100.0, f"Confidence {conf} out of range [50, 100] for logit {logit}"


# ==============================================================================
# 5. Invalid Image Handling & Robustness
# ==============================================================================

def test_invalid_image_corrupted_bytes():
    """Passing non-image bytes to PIL raises UnidentifiedImageError."""
    corrupted_stream = io.BytesIO(b"Not An Image File Header 1234567890")
    try:
        Image.open(corrupted_stream)
        assert False, "Expected UnidentifiedImageError was not raised."
    except UnidentifiedImageError:
        pass


def test_empty_image_stream():
    """Empty stream raises UnidentifiedImageError."""
    empty_stream = io.BytesIO(b"")
    try:
        Image.open(empty_stream)
        assert False, "Expected UnidentifiedImageError was not raised."
    except UnidentifiedImageError:
        pass


# ==============================================================================
# 6. End-to-End Prediction Contract (Mocked Model)
# ==============================================================================

def test_mocked_model_inference_pipeline():
    """Test full inference pipeline with mocked model predict to avoid heavy downloads."""
    mock_model = MagicMock()
    # Mocking a prediction returning logit 1.5
    mock_model.predict.return_value = np.array([[1.5]], dtype=np.float32)

    sample_img = Image.new("RGB", (200, 200), color=(100, 150, 200))
    processed = preprocess_image(sample_img)
    raw_prediction = mock_model.predict(processed, verbose=0)
    logit = float(raw_prediction[0][0])
    label, confidence = classify_logit(logit)

    assert label == "Dog"
    assert isinstance(confidence, float)
    assert 50.0 <= confidence <= 100.0
    mock_model.predict.assert_called_once()
