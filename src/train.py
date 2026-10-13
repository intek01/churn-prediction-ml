"""Trains and compares three classical classifiers on the same
preprocessed features: logistic regression (linear baseline), a single
decision tree (interpretable, usually overfits), and a random forest
(variance-reduced ensemble of trees). All three share the exact same
ColumnTransformer via sklearn Pipeline, so differences in the scores
reflect the model, not inconsistent preprocessing.
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .data_prep import TARGET
from .evaluate import score
from .features import add_tenure_bucket, build_preprocessor

RANDOM_STATE = 42

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "decision_tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced", random_state=RANDOM_STATE
    ),
}


def make_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE):
    df = add_tenure_bucket(df)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_and_evaluate(df: pd.DataFrame) -> dict:
    """Fits every model in MODELS on an identical train/test split and
    returns {model_name: metrics_dict}. class_weight="balanced" is used
    on all three instead of resampling, so the comparison stays on
    exactly the same rows for every model.
    """
    X_train, X_test, y_train, y_test = make_split(df)
    preprocessor = build_preprocessor(X_train)

    results = {}
    fitted = {}
    for name, estimator in MODELS.items():
        pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        results[name] = score(y_test, y_pred, y_proba)
        fitted[name] = pipeline

    return {"metrics": results, "models": fitted, "X_test": X_test, "y_test": y_test}


def top_feature_importances(pipeline: Pipeline, top_n: int = 10) -> list[tuple[str, float]]:
    """Only meaningful for the tree-based models (logistic regression's
    coefficients aren't directly comparable across one-hot columns on
    different scales, so this is not called for it).
    """
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    pairs = sorted(zip(feature_names, importances), key=lambda p: p[1], reverse=True)
    return pairs[:top_n]
