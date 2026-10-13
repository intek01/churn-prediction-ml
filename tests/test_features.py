import pandas as pd

from src.features import (
    NUMERIC_FEATURES,
    add_tenure_bucket,
    build_preprocessor,
    categorical_features,
)


def test_add_tenure_bucket_assigns_expected_bands():
    df = pd.DataFrame({"tenure": [0, 6, 12, 30, 60]})
    out = add_tenure_bucket(df)
    assert out["tenure_bucket"].tolist() == ["0-6mo", "0-6mo", "7-12mo", "25-48mo", "49mo+"]


def test_categorical_features_excludes_numeric_and_target():
    df = pd.DataFrame({**{c: [1] for c in NUMERIC_FEATURES}, "gender": ["Male"], "Churn": [0]})
    cats = categorical_features(df)
    assert cats == ["gender"]


def test_build_preprocessor_fits_and_transforms_mixed_types():
    df = pd.DataFrame(
        {
            "tenure": [1, 24], "MonthlyCharges": [50.0, 90.0],
            "TotalCharges": [50.0, 2160.0], "SeniorCitizen": [0, 1],
            "gender": ["Male", "Female"], "Contract": ["Month-to-month", "Two year"],
        }
    )
    pre = build_preprocessor(df)
    transformed = pre.fit_transform(df)
    assert transformed.shape[0] == 2
