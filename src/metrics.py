"""Computer vision metrics and confusion matrix formatter."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Computes Top-1 accuracy and classification performance."""
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        "accuracy_pct": round(acc * 100, 2),
        "confusion_matrix": cm,
        "classification_report": report,
    }


def print_confusion_matrix(cm: np.ndarray, labels: list = None) -> None:
    """Renders formatted ASCII confusion matrix table."""
    num_classes = cm.shape[0]
    if labels is None:
        labels = [str(i) for i in range(num_classes)]

    header = "True \\ Pred | " + " ".join(f"{lbl:>4}" for lbl in labels)
    print(header)
    print("-" * len(header))

    for idx, row in enumerate(cm):
        lbl = labels[idx]
        row_str = " ".join(f"{val:>4}" for val in row)
        print(f"{lbl:>10} | {row_str}")
