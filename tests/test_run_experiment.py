import sys

from src import run_experiment
from src.data_prep import clean
from src.train import MODELS


def test_main_prints_a_metrics_row_per_model(tmp_path, monkeypatch, capsys, synthetic_clean_df):
    # run_experiment.main() calls load_clean(path), i.e. it expects a RAW
    # csv (with customerID/Churn as "Yes"/"No"), so write one back out
    # from the already-cleaned synthetic fixture rather than re-deriving
    # a second synthetic generator just for this CLI-level test.
    raw_like = synthetic_clean_df.copy()
    raw_like["customerID"] = [f"T-{i}" for i in range(len(raw_like))]
    raw_like["Churn"] = raw_like["Churn"].map({1: "Yes", 0: "No"})
    csv_path = tmp_path / "raw.csv"
    raw_like.to_csv(csv_path, index=False)

    monkeypatch.setattr(sys, "argv", ["run_experiment", "--data", str(csv_path)])
    run_experiment.main()

    out = capsys.readouterr().out
    for model_name in MODELS:
        assert model_name in out
    assert "top random_forest feature importances" in out


def test_cleaning_a_cleaned_frame_is_a_no_op_on_total_charges(synthetic_clean_df):
    # Sanity check that the fixture used above really is already in the
    # shape clean() produces, so the CLI test above is testing the CLI
    # path and not accidentally double-cleaning.
    twice = clean(synthetic_clean_df.assign(customerID="X", Churn=synthetic_clean_df["Churn"].map({1: "Yes", 0: "No"})))
    assert twice["TotalCharges"].equals(synthetic_clean_df["TotalCharges"])
