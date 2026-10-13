import random

import pandas as pd
import pytest

from src.data_prep import clean

CATEGORICAL_OPTIONS = {
    "gender": ["Male", "Female"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
}


def _synthetic_raw_rows(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        tenure = rng.randint(0, 72)
        monthly = round(rng.uniform(18.0, 120.0), 2)
        row = {col: rng.choice(opts) for col, opts in CATEGORICAL_OPTIONS.items()}
        row.update(
            customerID=f"TEST-{i:04d}",
            SeniorCitizen=rng.choice([0, 1]),
            tenure=tenure,
            MonthlyCharges=monthly,
            TotalCharges=" " if tenure == 0 else f"{monthly * tenure:.2f}",
            # Deliberately correlated with tenure so the models have a
            # real (if crude) signal to find, rather than pure noise that
            # would make every metric meaningless by construction.
            Churn="Yes" if (tenure < 12 and rng.random() < 0.7) or rng.random() < 0.15 else "No",
        )
        rows.append(row)
    return rows


@pytest.fixture()
def synthetic_clean_df() -> pd.DataFrame:
    """A small (120-row) but structurally real dataset: every categorical
    column the real CSV has, a tenure-correlated churn label so the
    models aren't fitting pure noise, and the same blank-TotalCharges
    edge case for tenure == 0 customers. Not the real data (keeps tests
    fast and independent of the committed CSV) but shaped exactly like it.
    """
    raw = pd.DataFrame(_synthetic_raw_rows(120, seed=7))
    return clean(raw)
