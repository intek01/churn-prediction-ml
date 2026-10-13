"""Feature engineering and the preprocessing pipeline shared by every model.

Kept as a single ColumnTransformer (not per-model code) so all three
models in train.py see identical inputs — any difference in their scores
is about the model, not about inconsistent preprocessing.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_prep import TARGET

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


def categorical_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NUMERIC_FEATURES and c != TARGET]


def add_tenure_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Bucketing tenure (months as a customer) into contract-length-ish
    bands gives the tree models a coarser, more stable split point than
    raw month counts, and gives logistic regression a non-linear signal
    it can't otherwise see from a single numeric column.
    """
    df = df.copy()
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, 10_000],
        labels=["0-6mo", "7-12mo", "13-24mo", "25-48mo", "49mo+"],
    ).astype(str)
    return df


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    cat_cols = categorical_features(df)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
