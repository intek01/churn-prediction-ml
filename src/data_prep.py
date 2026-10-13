"""Load and clean the IBM Telco Customer Churn dataset.

TotalCharges is read as a string in the raw CSV because 11 rows (all new
customers with tenure == 0) have it blank instead of "0.00" — a real
artifact of this dataset, not a synthetic one. Those rows are imputed to
0.0 rather than dropped, since tenure == 0 fully explains the blank and
dropping customers is a bigger distortion than a documented zero-fill.
"""
from __future__ import annotations

import pandas as pd

TARGET = "Churn"
ID_COLUMN = "customerID"


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges arrives as a string column because of the blank entries
    # (" "); to_numeric coerces those (and anything else unparseable) to
    # NaN, which fillna then zeroes — same outcome as before, but without
    # assigning a float into a pandas string-dtype column in place.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # SeniorCitizen is 0/1 int in the raw file; keep it numeric, not a
    # Yes/No string, so downstream encoding treats it consistently with
    # the other binary numeric flag instead of as a separate category.
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    df[TARGET] = (df[TARGET] == "Yes").astype(int)

    return df.drop(columns=[ID_COLUMN])


def load_clean(path: str) -> pd.DataFrame:
    return clean(load_raw(path))
