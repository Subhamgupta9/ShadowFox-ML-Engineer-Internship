"""
Inference and image processing module for Task 1: Image Tagging with TensorFlow.
Provides reusable model loading, preprocessing, logit-to-probability mapping, and classification.
"""

from pathlib import Path
from typing import Tuple, Union, Optional
import numpy as np
from PIL import Image

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "shadowfox_cats_dogs_mobilenetv2.keras"
TARGET_SIZE = (160, 160)

def preprocess_image(image: Image.Image, target_size: Tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """
    Validates and preprocesses an input PIL image for MobileNetV2 classification.

    Converts image to RGB mode, resizes to target_size (default 160x160), converts to float32,
    and adds a batch dimension.

    Args:
        image: A PIL.Image.Image instance.
        target_size: Desired (width, height) tuple.

    Returns:
        np.ndarray with shape (1, height, width, 3) and dtype float32.

    Raises:
        ValueError: If image is None, not a PIL Image, or has empty dimensions.
    """
    if image is None:
        raise ValueError("Input image cannot be None.")
    if not isinstance(image, Image.Image):
        raise ValueError(f"Expected PIL Image, got {type(image).__name__}")
    if image.width <= 0 or image.height <= 0:
        raise ValueError("Image dimensions must be positive non-zero values.")

    rgb_image = image.convert("RGB")
    resized_image = rgb_image.resize(target_size)
    image_array = np.expand_dims(np.array(resized_image, dtype=np.float32), axis=0)
    return image_array

def logit_to_probability(raw_logit: float) -> float:
    """
    Computes numerically stable sigmoid activation for a raw scalar logit.
    1 / (1 + exp(-logit))
    """
    # Numerically stable sigmoid
    if raw_logit >= 0:
        z = np.exp(-raw_logit)
        return float(1.0 / (1.0 + z))
    else:
        z = np.exp(raw_logit)
        return float(z / (1.0 + z))

def classify_prediction(dog_probability: float) -> Tuple[str, float]:
    """
    Maps a dog class probability in [0.0, 1.0] to a class label and percentage confidence.

    Decision rule:
    - If dog_probability >= 0.5: Class is "Dog", confidence is dog_probability * 100.
    - If dog_probability < 0.5: Class is "Cat", confidence is (1 - dog_probability) * 100.

    Confidence is strictly bounded in [50.0, 100.0]%.

    Args:
        dog_probability: Probability that the image contains a dog, in [0.0, 1.0].

    Returns:
        Tuple of (label, confidence_percentage) where label is 'Dog' or 'Cat'.
    """
    if not (0.0 <= dog_probability <= 1.0):
        raise ValueError(f"Probability must be between 0.0 and 1.0, got {dog_probability}")

    if dog_probability >= 0.5:
        return "Dog", float(dog_probability * 100.0)
    else:
        return "Cat", float((1.0 - dog_probability) * 100.0)

def load_classifier(model_path: Optional[Union[str, Path]] = None):
    """
    Loads the Keras MobileNetV2 classifier with custom preprocess_input object.

    Args:
        model_path: Path to the .keras model file (defaults to DEFAULT_MODEL_PATH).

    Returns:
        Loaded tf.keras.Model.
    """
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")

    return tf.keras.models.load_model(
        str(path),
        custom_objects={"preprocess_input": preprocess_input},
        compile=False
    )

def predict_image(model, image: Image.Image) -> Tuple[str, float]:
    """
    End-to-end inference function for a single PIL image.

    Args:
        model: Loaded Keras model (or mock estimator responding to .predict).
        image: Input PIL Image.

    Returns:
        Tuple of (label, confidence_percentage).
    """
    image_array = preprocess_image(image)
    raw_pred = model.predict(image_array, verbose=0)
    raw_logit = float(raw_pred[0][0])
    dog_prob = logit_to_probability(raw_logit)
    return classify_prediction(dog_prob)
