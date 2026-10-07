"""Model evaluation module.

Computes standard ML evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
"""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def compute_metrics(y_true, y_pred, average: str = "binary", pos_label=None) -> Dict[str, Any]:
    """Computes standard evaluation metrics for classification.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging method for multi-class ('binary', 'macro', 'weighted')
        pos_label: Positive class label if binary

    Returns:
        Dictionary containing accuracy, precision, recall, f1, confusion matrix,
        and full classification report text.
    """
    labels = np.unique(y_true)

    # Determine average type if pos_label is binary vs multiclass
    if len(labels) == 2 and pos_label is None:
        # Default pos_label to either 'Fake' or 1 if present
        if "Fake" in labels:
            pos_label = "Fake"
        elif 1 in labels:
            pos_label = 1
        else:
            pos_label = labels[0]

    # Compute metrics safely
    acc = accuracy_score(y_true, y_pred)

    if len(labels) == 2 and pos_label is not None:
        prec = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    else:
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, zero_division=0)
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "labels": [str(lbl) for lbl in labels],
        "classification_report": report,
        "classification_report_dict": report_dict
    }


def format_metrics_summary(metrics: Dict[str, Any]) -> str:
    """Formats evaluation metrics into an easy-to-read viva summary string."""
    summary = (
        f"Evaluation Metrics Summary:\n"
        f"--------------------------\n"
        f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)\n"
        f"Precision: {metrics['precision']:.4f}\n"
        f"Recall:    {metrics['recall']:.4f}\n"
        f"F1-Score:  {metrics['f1_score']:.4f}\n"
    )
    return summary
