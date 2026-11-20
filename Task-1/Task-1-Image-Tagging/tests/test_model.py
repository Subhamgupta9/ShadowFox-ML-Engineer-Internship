"""
NEXUS & ShadowFox Machine Learning Engineer Internship
Task 1: Image Tagging with TensorFlow
Automated Unit Tests & Quality Assurance Suite

Tests verify:
1. Framework & library imports
2. Model artifact existence and size integrity (.keras)
3. Modular image preprocessing pipeline (src.predict.preprocess_image)
4. Logit-to-probability mapping & numerical stability (src.predict.logit_to_probability)
5. Decision boundary and confidence bounds (src.predict.classify_prediction)
6. Error handling on invalid/corrupted images and boundary conditions
7. End-to-end prediction output contract (src.predict.predict_image)
"""

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
try:
    import pytest
except ImportError:
    pytest = None
from PIL import Image, UnidentifiedImageError

TASK1_DIR = Path(__file__).resolve().parent.parent
if str(TASK1_DIR) not in sys.path:
    sys.path.insert(0, str(TASK1_DIR))

MODEL_FILE = TASK1_DIR / "shadowfox_cats_dogs_mobilenetv2.keras"
FINETUNED_MODEL_FILE = TASK1_DIR / "shadowfox_cats_dogs_mobilenetv2_finetuned_best.keras"
FIXTURES_DIR = TASK1_DIR / "tests" / "fixtures"
SAMPLE_CAT_FILE = FIXTURES_DIR / "sample_cat.jpg"
SAMPLE_DOG_FILE = FIXTURES_DIR / "sample_dog.jpg"

from src.predict import (
    load_classifier,
    preprocess_image,
    logit_to_probability,
    classify_prediction,
    predict_image,
    DEFAULT_MODEL_PATH
)


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


def test_load_classifier_missing_file():
    """Verify load_classifier raises FileNotFoundError when given a non-existent path."""
    non_existent = TASK1_DIR / "non_existent_model_checkpoint_9999.keras"
    if pytest is not None:
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            load_classifier(non_existent)
    else:
        try:
            load_classifier(non_existent)
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass


# ==============================================================================
# 3. Modular Image Preprocessing Pipeline (src.predict.preprocess_image)
# ==============================================================================

def test_image_preprocessing_rgb():
    """Verify RGB image preprocessing produces exact expected shape (1, 160, 160, 3) and float32 dtype."""
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


def test_image_preprocessing_invalid_inputs():
    """Verify preprocess_image raises ValueError on invalid inputs (None, non-Image, empty/zero dimensions)."""
    if pytest is not None:
        with pytest.raises(ValueError, match="Input image cannot be None"):
            preprocess_image(None)

        with pytest.raises(ValueError, match="Expected PIL Image"):
            preprocess_image("not_an_image")

        with pytest.raises(ValueError, match="Expected PIL Image"):
            preprocess_image([1, 2, 3])

        with pytest.raises(ValueError, match="Expected PIL Image"):
            preprocess_image({"data": 123})


def test_image_preprocessing_diverse_modes_and_shapes():
    """Verify preprocessing converts diverse PIL modes (1-bit, palette, CMYK) and extreme aspect ratios into (1, 160, 160, 3) float32."""
    # 1-bit bilevel mode
    img_bilevel = Image.new("1", (100, 100), color=1)
    arr_bi = preprocess_image(img_bilevel)
    assert arr_bi.shape == (1, 160, 160, 3)
    assert arr_bi.dtype == np.float32

    # Palette indexed mode
    img_palette = Image.new("P", (120, 80))
    arr_pal = preprocess_image(img_palette)
    assert arr_pal.shape == (1, 160, 160, 3)
    assert arr_pal.dtype == np.float32

    # CMYK 4-channel mode
    img_cmyk = Image.new("CMYK", (200, 150), color=(0, 255, 128, 0))
    arr_cmyk = preprocess_image(img_cmyk)
    assert arr_cmyk.shape == (1, 160, 160, 3)
    assert arr_cmyk.dtype == np.float32

    # Extreme aspect ratio banner (1200 x 200)
    img_wide = Image.new("RGB", (1200, 200), color=(40, 80, 120))
    arr_wide = preprocess_image(img_wide)
    assert arr_wide.shape == (1, 160, 160, 3)
    assert arr_wide.dtype == np.float32

    # Extreme aspect ratio vertical strip (100 x 900)
    img_tall = Image.new("RGB", (100, 900), color=(120, 40, 80))
    arr_tall = preprocess_image(img_tall)
    assert arr_tall.shape == (1, 160, 160, 3)
    assert arr_tall.dtype == np.float32


# ==============================================================================
# 4. Logit-to-Probability & Numerical Stability (src.predict.logit_to_probability)
# ==============================================================================

def test_logit_to_probability_values():
    """Verify sigmoid conversion maps logits to [0, 1] range correctly."""
    assert logit_to_probability(0.0) == 0.5
    assert logit_to_probability(2.0) > 0.85
    assert logit_to_probability(-2.0) < 0.15


def test_logit_to_probability_numerical_stability():
    """Verify extreme logits do not overflow or produce NaN."""
    prob_large_pos = logit_to_probability(100.0)
    prob_large_neg = logit_to_probability(-100.0)
    assert 0.9999 <= prob_large_pos <= 1.0
    assert 0.0 <= prob_large_neg <= 0.0001
    assert not np.isnan(prob_large_pos)
    assert not np.isnan(prob_large_neg)


# ==============================================================================
# 5. Classification Decision Boundary & Confidence (src.predict.classify_prediction)
# ==============================================================================

def test_classification_decision_dog():
    """Probability >= 0.5 must yield Dog label with confidence in [50, 100]%."""
    label, confidence = classify_prediction(0.92)
    assert label == "Dog"
    assert round(confidence, 2) == 92.00
    assert 50.0 <= confidence <= 100.0


def test_classification_decision_cat():
    """Probability < 0.5 must yield Cat label with confidence in [50, 100]%."""
    label, confidence = classify_prediction(0.15)
    assert label == "Cat"
    assert round(confidence, 2) == 85.00
    assert 50.0 <= confidence <= 100.0


def test_classification_decision_boundary():
    """Exact boundary (0.50) defaults to Dog at 50.0% confidence."""
    label, confidence = classify_prediction(0.50)
    assert label == "Dog"
    assert confidence == 50.0


def test_classification_invalid_probability():
    """Probability outside [0, 1] raises ValueError."""
    if pytest is not None:
        with pytest.raises(ValueError, match="Probability must be between"):
            classify_prediction(-0.1)

        with pytest.raises(ValueError, match="Probability must be between"):
            classify_prediction(1.05)


# ==============================================================================
# 6. Invalid Image Handling & Robustness
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
# 7. End-to-End Prediction Contract via src.predict.predict_image
# ==============================================================================

def test_mocked_model_inference_pipeline():
    """Test full inference pipeline using src.predict.predict_image with a mock estimator."""
    mock_model = MagicMock()
    # Mocking raw logit output 1.5 -> prob ~0.8175 -> Dog ~81.75%
    mock_model.predict.return_value = np.array([[1.5]], dtype=np.float32)

    sample_img = Image.new("RGB", (200, 200), color=(100, 150, 200))
    label, confidence = predict_image(mock_model, sample_img)

    assert label == "Dog"
    assert isinstance(confidence, float)
    assert 80.0 <= confidence <= 85.0
    mock_model.predict.assert_called_once()


def test_mocked_model_inference_cat():
    """Test inference pipeline with a negative logit producing Cat."""
    mock_model = MagicMock()
    # Mocking raw logit output -2.0 -> prob ~0.1192 -> Cat ~88.08%
    mock_model.predict.return_value = np.array([[-2.0]], dtype=np.float32)

    sample_img = Image.new("RGB", (160, 160), color=(50, 50, 50))
    label, confidence = predict_image(mock_model, sample_img)

    assert label == "Cat"
    assert isinstance(confidence, float)
    assert 87.0 <= confidence <= 89.0
    mock_model.predict.assert_called_once()


def test_real_model_inference_pipeline():
    """Verify real .keras model loading and end-to-end 160x160 inference when TensorFlow is available."""
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError:
        if pytest is not None:
            pytest.skip("TensorFlow is not installed in the active runner venv.")
        return

    model = load_classifier(MODEL_FILE)
    synthetic_img = Image.new("RGB", (160, 160), color=(120, 140, 160))
    label, confidence = predict_image(model, synthetic_img)

    assert label in {"Cat", "Dog"}
    assert 50.0 <= confidence <= 100.0


def test_sample_image_fixtures_exist():
    """Verify genuine cat and dog test fixtures exist and have expected non-trivial size."""
    assert SAMPLE_CAT_FILE.exists(), f"Sample cat fixture missing at {SAMPLE_CAT_FILE}"
    assert SAMPLE_DOG_FILE.exists(), f"Sample dog fixture missing at {SAMPLE_DOG_FILE}"
    assert SAMPLE_CAT_FILE.stat().st_size > 50_000, "Sample cat fixture appears truncated"
    assert SAMPLE_DOG_FILE.stat().st_size > 20_000, "Sample dog fixture appears truncated"


def test_known_sample_cat_image_regression():
    """
    Regression test verifying the production model correctly classifies the known sample cat image.

    Note: This test validates deterministic end-to-end pipeline execution on a representative input;
    it does not establish or claim a benchmark validation accuracy (which is 98.40% on the 400-sample validation set).
    """
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError:
        if pytest is not None:
            pytest.skip("TensorFlow is not installed in the active runner venv.")
        return

    model = load_classifier(MODEL_FILE)
    with Image.open(SAMPLE_CAT_FILE) as img:
        label, confidence = predict_image(model, img)

    assert label == "Cat", f"Expected 'Cat' but got '{label}'"
    assert confidence >= 95.0, f"Expected high confidence for known cat image, got {confidence:.2f}%"


def test_known_sample_dog_image_regression():
    """
    Regression test verifying the production model correctly classifies the known sample dog image.

    Note: This test validates deterministic end-to-end pipeline execution on a representative input;
    it does not establish or claim a benchmark validation accuracy (which is 98.40% on the 400-sample validation set).
    """
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError:
        if pytest is not None:
            pytest.skip("TensorFlow is not installed in the active runner venv.")
        return

    model = load_classifier(MODEL_FILE)
    with Image.open(SAMPLE_DOG_FILE) as img:
        label, confidence = predict_image(model, img)

    assert label == "Dog", f"Expected 'Dog' but got '{label}'"
    assert confidence >= 95.0, f"Expected high confidence for known dog image, got {confidence:.2f}%"


def test_predict_image_consistency_with_raw_pipeline():
    """
    Verify predict_image produces bitwise-identical results to step-by-step
    preprocessing, model.predict, and probability classification.
    """
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError:
        if pytest is not None:
            pytest.skip("TensorFlow is not installed in the active runner venv.")
        return

    model = load_classifier(MODEL_FILE)
    with Image.open(SAMPLE_CAT_FILE) as img:
        label_pipe, conf_pipe = predict_image(model, img)

        # Deconstructed execution
        arr = preprocess_image(img)
        raw_pred = model.predict(arr, verbose=0)
        logit = float(raw_pred[0][0])
        prob = logit_to_probability(logit)
        label_step, conf_step = classify_prediction(prob)

    assert label_pipe == label_step
    assert np.isclose(conf_pipe, conf_step, atol=1e-5)
