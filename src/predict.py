"""Inference module for Fake Product Review Detection.

Loads the trained TF-IDF vectorizer and Multinomial Naive Bayes model
to predict whether a review is Likely Fake or Likely Genuine.
If artifacts are missing, clearly reports that the model is not trained yet.
"""

import os
import joblib
from typing import Dict, Any, Optional

from src.preprocessing import clean_text, extract_review_heuristics

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")

LABEL_MAPPING = {
    "CG": "Fake",
    "OR": "Genuine"
}

_CACHED_MODEL = None
_CACHED_VECTORIZER = None


def is_model_trained() -> bool:
    """Checks whether the required model artifacts exist on disk."""
    return os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)


def load_model_artifacts():
    """Loads and caches (model, vectorizer) if present, else (None, None)."""
    global _CACHED_MODEL, _CACHED_VECTORIZER
    if not is_model_trained():
        return None, None
    if _CACHED_MODEL is None or _CACHED_VECTORIZER is None:
        _CACHED_MODEL = joblib.load(MODEL_PATH)
        _CACHED_VECTORIZER = joblib.load(VECTORIZER_PATH)
    return _CACHED_MODEL, _CACHED_VECTORIZER


def predict_review(raw_review: str) -> Dict[str, Any]:
    """Analyzes a customer review using the trained NLP model and heuristics.

    IMPORTANT: Does not hardcode or fabricate predictions.
    If model is not trained, returns status="not_trained".

    Args:
        raw_review: User input review text

    Returns:
        Dictionary containing:
        - status: "success", "empty_input", or "not_trained"
        - prediction_label: e.g. "Likely Fake Review" or "Likely Genuine Review"
        - predicted_class: "Fake" or "Genuine"
        - confidence: Likelihood/confidence percentage
        - class_probabilities: Dict mapping class names to likelihood percentages
        - cleaned_text: Review after NLP preprocessing
        - heuristics: Supporting indicators (word count, suspicious words, etc.)
    """
    if not is_model_trained():
        return {
            "status": "not_trained",
            "message": "Model not trained yet. Please train the model using a dataset first."
        }

    if not raw_review or not raw_review.strip():
        return {
            "status": "empty_input",
            "message": "Please enter a valid review text."
        }

    # 1. Linguistic heuristics for supporting context
    heuristics = extract_review_heuristics(raw_review)

    # 2. NLP Preprocessing (identical to training)
    cleaned = clean_text(raw_review)
    if not cleaned:
        # Fallback if text contained only punctuation or stopwords
        cleaned = raw_review.lower().strip()

    # 3. Model Prediction
    model, vectorizer = load_model_artifacts()
    if model is None or vectorizer is None:
        return {
            "status": "not_trained",
            "message": "Failed to load model artifacts."
        }

    text_vectorized = vectorizer.transform([cleaned])
    raw_pred = model.predict(text_vectorized)[0]

    # Standardize output class
    raw_pred_str = str(raw_pred).strip()
    standard_class = LABEL_MAPPING.get(raw_pred_str, raw_pred_str)

    # Get class probabilities if available
    class_probabilities = {}
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(text_vectorized)[0]
        for cls_name, prob in zip(model.classes_, proba):
            mapped_cls = LABEL_MAPPING.get(str(cls_name), str(cls_name))
            class_probabilities[mapped_cls] = round(float(prob) * 100, 2)
        confidence = round(float(max(proba)) * 100, 2)

    # Human-friendly label
    if standard_class.lower() in ("fake", "cg"):
        normalized_label = "Likely Fake Review"
        is_fake = True
    else:
        normalized_label = "Likely Genuine Review"
        is_fake = False

    return {
        "status": "success",
        "raw_prediction": raw_pred_str,
        "predicted_class": standard_class,
        "prediction_label": normalized_label,
        "is_fake": is_fake,
        "confidence": confidence if confidence else None,
        "class_probabilities": class_probabilities,
        "cleaned_text": cleaned,
        "heuristics": heuristics
    }
