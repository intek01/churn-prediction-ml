import pandas as pd

from src.data_prep import TARGET, clean, load_clean, load_raw

RAW_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]


def make_raw_row(**overrides) -> dict:
    row = {col: "placeholder" for col in RAW_COLUMNS}
    row.update(
        customerID="CUST-1", SeniorCitizen=0, tenure=5,
        MonthlyCharges=70.5, TotalCharges="352.50", Churn="No",
    )
    row.update(overrides)
    return row


def test_blank_total_charges_is_imputed_to_zero():
    df = pd.DataFrame([make_raw_row(tenure=0, TotalCharges=" ")])
    cleaned = clean(df)
    assert cleaned["TotalCharges"].iloc[0] == 0.0


def test_total_charges_is_numeric_after_cleaning():
    df = pd.DataFrame([make_raw_row(TotalCharges="352.50")])
    cleaned = clean(df)
    assert cleaned["TotalCharges"].dtype.kind == "f"
    assert cleaned["TotalCharges"].iloc[0] == 352.50


def test_churn_is_encoded_as_binary():
    df = pd.DataFrame([make_raw_row(Churn="Yes"), make_raw_row(Churn="No")])
    cleaned = clean(df)
    assert cleaned[TARGET].tolist() == [1, 0]


def test_customer_id_is_dropped():
    df = pd.DataFrame([make_raw_row()])
    cleaned = clean(df)
    assert "customerID" not in cleaned.columns


def test_senior_citizen_stays_numeric():
    df = pd.DataFrame([make_raw_row(SeniorCitizen=1)])
    cleaned = clean(df)
    assert cleaned["SeniorCitizen"].iloc[0] == 1
    assert cleaned["SeniorCitizen"].dtype.kind in "iu"


def test_load_raw_reads_the_committed_csv():
    df = load_raw("data/raw/Telco-Customer-Churn.csv")
    assert len(df) == 7043
    assert "Churn" in df.columns


def test_load_clean_end_to_end_on_the_real_dataset():
    df = load_clean("data/raw/Telco-Customer-Churn.csv")
    assert "customerID" not in df.columns
    assert set(df[TARGET].unique()) == {0, 1}
    assert df["TotalCharges"].isna().sum() == 0
    assert df["TotalCharges"].dtype.kind == "f"
