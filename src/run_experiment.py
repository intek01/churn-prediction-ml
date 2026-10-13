"""Standalone CLI: loads the data, trains all three models, prints a
comparison table and the random forest's top feature importances.

    python -m src.run_experiment --data data/raw/Telco-Customer-Churn.csv
"""
from __future__ import annotations

import argparse

from .data_prep import load_clean
from .train import top_feature_importances, train_and_evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/Telco-Customer-Churn.csv")
    args = parser.parse_args()

    df = load_clean(args.data)
    result = train_and_evaluate(df)

    print(f"{'model':<22}{'accuracy':>10}{'precision':>12}{'recall':>10}{'f1':>10}{'roc_auc':>10}")
    for name, m in result["metrics"].items():
        print(
            f"{name:<22}{m['accuracy']:>10.3f}{m['precision']:>12.3f}"
            f"{m['recall']:>10.3f}{m['f1']:>10.3f}{m['roc_auc']:>10.3f}"
        )

    print("\ntop random_forest feature importances:")
    for feat, importance in top_feature_importances(result["models"]["random_forest"]):
        print(f"  {feat:<35}{importance:.4f}")


if __name__ == "__main__":
    main()
