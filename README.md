# churn-prediction-ml

A classical supervised-learning project: predicting customer churn on the
[IBM Telco Customer Churn dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
(7,043 customers, 21 columns — a public dataset IBM publishes for exactly
this kind of churn-modeling exercise, committed here under `data/raw/`).
Three models — logistic regression, a decision tree, and a random
forest — are trained on identical preprocessed features and compared
head-to-head, rather than reporting a single model's number in isolation.

## Why this exists

Most of this portfolio's projects (`pdf-qa-agent`, `erp-ops-agent`,
`sales-etl-pipeline`) are LLM-agent or data-engineering work. This one is
deliberately the opposite: no LLMs, no orchestration — just loading real
tabular data, engineering features from it, fitting a few standard
classifiers, and evaluating them honestly on a problem (churn) where the
classes are imbalanced enough that a naive model looks deceptively good.

## What it does

- **`src/data_prep.py`** — loads the raw CSV and cleans it: encodes
  `Churn` to 0/1, fixes the 11 rows where `TotalCharges` is blank (all
  brand-new customers with `tenure == 0` — a real quirk of this dataset,
  not injected), drops the customer ID.
- **`src/features.py`** — a single `ColumnTransformer` (standard-scaled
  numerics + one-hot categoricals) shared by every model, plus a derived
  `tenure_bucket` feature, so score differences between models reflect
  the model, not inconsistent preprocessing.
- **`src/train.py`** — fits `LogisticRegression`, `DecisionTreeClassifier`,
  and `RandomForestClassifier` (all with `class_weight="balanced"`) on
  one stratified 80/20 train/test split, and exposes random-forest
  feature importances.
- **`src/evaluate.py`** — precision, recall, F1, and ROC-AUC. Accuracy is
  computed but deliberately not the headline number (see below).
- **`src/run_experiment.py`** — CLI: `python -m src.run_experiment`.

## Results (actual run against the committed dataset, 80/20 split, seed 42)

| model | accuracy | precision | recall | f1 | roc_auc |
|---|---|---|---|---|---|
| logistic_regression | 0.739 | 0.505 | 0.802 | 0.620 | 0.845 |
| decision_tree | 0.740 | 0.507 | 0.789 | 0.617 | 0.828 |
| random_forest | 0.756 | 0.528 | 0.759 | 0.623 | 0.840 |

**Why accuracy isn't the headline metric:** churn is 26.5% of this
dataset. A model that always predicts "will not churn" scores ~73.5%
accuracy while catching zero actual churners — useless for the business
problem. That's why `class_weight="balanced"` is used on every model (so
the minority class isn't ignored) and why recall/F1/ROC-AUC are reported
alongside accuracy rather than instead of it.

Logistic regression has the best ROC-AUC and the highest recall, catching
80% of churners at the cost of more false positives (lower precision).
Random forest trades some recall for precision and the best accuracy.
Which trade-off is "better" depends on whether a false positive (offering
a retention incentive to someone who wasn't leaving) or a false negative
(losing a customer with no intervention) costs more — a business decision
this project surfaces rather than resolves.

**Top random-forest feature importances:** `Contract_Month-to-month`,
`tenure`, `TotalCharges`, `MonthlyCharges`, `Contract_Two year` —
month-to-month customers with short tenure and lower lifetime spend are
the clearest churn signal, which also matches the univariate pattern in
the raw data (new customers haven't yet committed to a longer contract).

## Tests

```bash
pip install -r requirements.txt
pytest -v --cov=src --cov-report=term-missing
```

**18 tests, 99% coverage** (86 statements, 1 missed — the
`if __name__ == "__main__":` guard). Ruff-clean. Most tests run against a
small synthetic fixture shaped like the real data (for speed and
independence from the committed CSV); `test_data_prep.py` additionally
runs `load_raw`/`load_clean` against the actual committed dataset end to
end, so the real file's edge cases (the blank `TotalCharges` rows) are
genuinely exercised, not just assumed.

## What this project does not claim

- No hyperparameter tuning was done — `max_depth`/`n_estimators` are
  reasonable defaults, not the result of a grid search. The point here is
  a fair comparison across model families, not squeezing out the last
  percentage point.
- Class imbalance is handled via `class_weight="balanced"`, not
  resampling (SMOTE, undersampling) — a deliberate, simpler choice,
  documented here rather than left unstated.
- This is tabular, classical ML — no deep learning. (That gap is covered
  elsewhere in this portfolio by a separate research publication on an
  LSTM-XGBoost ensemble, which is a different kind of problem — time-series
  forecasting, not classification.)
