"""Evaluation metrics. Accuracy is deliberately not the headline metric —
with a 73/27 class split, a model that always predicts "no churn" scores
~73% accuracy while catching zero actual churners, which is exactly the
failure mode a churn model must not have.
"""
from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def score(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }
