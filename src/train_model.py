"""Model training pipeline for Fake Product Review Detection.

Pipeline Steps:
1. Load dataset (using src.data_loader with text deduplication)
2. Clean text using the exact same preprocessing (src.preprocessing.clean_text)
3. Map labels clearly (CG -> Fake, OR -> Genuine)
4. Split into training and testing sets (80% train, 20% test, stratified, random_state=42)
5. Transform text with TF-IDF Vectorizer (fit ONLY on training data)
6. Train Multinomial Naive Bayes Classifier
7. Evaluate model on unseen test set (src.evaluate.compute_metrics)
8. Save model.pkl, vectorizer.pkl, and metrics.json in models/
"""

import os
import json
import joblib
from typing import Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from src.data_loader import load_dataset, get_available_datasets
from src.preprocessing import clean_text
from src.evaluate import compute_metrics, format_metrics_summary

# Default Paths & Constants
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

LABEL_MAPPING = {
    "CG": "Fake",
    "OR": "Genuine"
}


def get_default_dataset_path() -> str:
    """Returns the primary dataset path if found, or first available in dataset/."""
    primary = os.path.join("dataset", "fake reviews dataset.csv")
    if os.path.exists(primary):
        return primary
    available = get_available_datasets("dataset")
    if available:
        return os.path.join("dataset", available[0])
    return os.path.join("dataset", "fake reviews dataset.csv")


def train(
    dataset_path: Optional[str] = None,
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 25000,
    min_df: int = 3,
    ngram_range: tuple = (1, 2)
) -> Dict[str, Any]:
    """Trains TF-IDF + Multinomial Naive Bayes model on the real dataset."""
    if not dataset_path:
        dataset_path = get_default_dataset_path()

    print("=" * 65)
    print("AI-Powered Fake Product Review Detection - ML Training Pipeline")
    print("=" * 65)

    # 1. Load Data
    print(f"\n[1/7] Loading dataset from: {dataset_path}...")
    df, text_col, label_col = load_dataset(
        filepath=dataset_path,
        text_column=text_column,
        label_column=label_column,
        drop_duplicates=True  # Prevent duplicate reviews across train/test splits
    )
    print(f" -> Found {len(df)} unique records after deduplication.")
    print(f" -> Text column: '{text_col}' | Label column: '{label_col}'")

    # 2. Map Labels
    print("\n[2/7] Mapping target labels...")
    raw_unique = df[label_col].unique().tolist()
    print(f" -> Detected raw labels: {raw_unique}")
    df["mapped_label"] = df[label_col].astype(str).str.strip().map(
        lambda x: LABEL_MAPPING.get(x, x)
    )
    mapped_dist = df["mapped_label"].value_counts().to_dict()
    print(f" -> Label mapping applied: {LABEL_MAPPING}")
    print(f" -> Mapped distribution: {mapped_dist}")

    # 3. Preprocess Text
    print("\n[3/7] Applying NLP preprocessing (lowercase, URLs, punctuation, stopwords)...")
    cleaned_texts = df[text_col].astype(str).apply(clean_text)
    labels = df["mapped_label"]

    # Filter out any records that are empty after text normalization
    valid_mask = cleaned_texts.str.strip() != ""
    cleaned_texts = cleaned_texts[valid_mask].reset_index(drop=True)
    labels = labels[valid_mask].reset_index(drop=True)
    print(f" -> {len(cleaned_texts)} records ready after NLP text cleaning.")

    # 4. Train-Test Split (Stratified, 80% train / 20% test)
    print(f"\n[4/7] Splitting dataset ({100 - int(test_size * 100)}% train, {int(test_size * 100)}% test, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    print(f" -> Training samples: {len(X_train)} ({len(X_train)/len(cleaned_texts)*100:.1f}%)")
    print(f" -> Test samples:     {len(X_test)} ({len(X_test)/len(cleaned_texts)*100:.1f}%)")

    # 5. Feature Extraction: TF-IDF (Fit ONLY on training data to prevent data leakage)
    print(f"\n[5/7] Fitting TF-IDF Vectorizer on training data only...")
    print(f"      Parameters: ngram_range={ngram_range}, min_df={min_df}, max_features={max_features}")
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    num_features = len(vectorizer.vocabulary_)
    print(f" -> TF-IDF Vocabulary Size: {num_features} features")

    # 6. Model Training: Multinomial Naive Bayes
    print("\n[6/7] Training Multinomial Naive Bayes Classifier (alpha=1.0)...")
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_vec, y_train)
    print(" -> Model fitting completed.")

    # 7. Evaluation on Unseen Test Set
    print("\n[7/7] Evaluating model on unseen test set...")
    y_pred = model.predict(X_test_vec)
    metrics = compute_metrics(y_test, y_pred, pos_label="Fake")

    print("\n" + "=" * 65)
    print(format_metrics_summary(metrics))
    print(f"Confusion Matrix (labels={metrics['labels']}):")
    for row in metrics["confusion_matrix"]:
        print(f"  {row}")
    print("\nClassification Report:\n" + metrics["classification_report"])
    print("=" * 65)

    # 8. Save Model Artifacts & Evaluation Metrics
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    metrics_payload = {
        "model_type": "Multinomial Naive Bayes",
        "feature_extractor": "TF-IDF Vectorizer",
        "dataset_file": os.path.basename(dataset_path),
        "total_valid_samples": len(cleaned_texts),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "tfidf_features": num_features,
        "ngram_range": list(ngram_range),
        "min_df": min_df,
        "max_features": max_features,
        "label_mapping": LABEL_MAPPING,
        "labels": metrics["labels"],
        "accuracy": round(metrics["accuracy"], 4),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "f1_score": round(metrics["f1_score"], 4),
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report"],
        "classification_report_dict": metrics.get("classification_report_dict", {})
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\nArtifacts successfully created and saved:")
    print(f" -> Model:      {MODEL_PATH}")
    print(f" -> Vectorizer: {VECTORIZER_PATH}")
    print(f" -> Metrics:    {METRICS_PATH}")
    print("=" * 65)

    return metrics_payload


if __name__ == "__main__":
    import sys
    cli_path = sys.argv[1] if len(sys.argv) > 1 else None
    train(dataset_path=cli_path)
