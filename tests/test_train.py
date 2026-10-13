from src.evaluate import score
from src.train import MODELS, make_split, top_feature_importances, train_and_evaluate


def test_make_split_is_stratified(synthetic_clean_df):
    _, X_test, y_train, y_test = make_split(synthetic_clean_df, test_size=0.25)
    assert len(X_test) == len(y_test)
    assert abs(y_train.mean() - y_test.mean()) < 0.15  # stratify keeps class ratios close


def test_train_and_evaluate_runs_every_model(synthetic_clean_df):
    result = train_and_evaluate(synthetic_clean_df)
    assert set(result["metrics"].keys()) == set(MODELS.keys())
    for metrics in result["metrics"].values():
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            assert 0.0 <= metrics[key] <= 1.0


def test_fitted_pipelines_predict_on_held_out_rows(synthetic_clean_df):
    result = train_and_evaluate(synthetic_clean_df)
    preds = result["models"]["random_forest"].predict(result["X_test"])
    assert len(preds) == len(result["y_test"])
    assert set(preds.tolist()) <= {0, 1}


def test_top_feature_importances_returns_sorted_pairs(synthetic_clean_df):
    result = train_and_evaluate(synthetic_clean_df)
    pairs = top_feature_importances(result["models"]["random_forest"], top_n=5)
    assert len(pairs) == 5
    importances = [v for _, v in pairs]
    assert importances == sorted(importances, reverse=True)


def test_score_matches_a_known_perfect_prediction():
    perfect = score([1, 0, 1, 0], [1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert perfect["accuracy"] == 1.0
    assert perfect["roc_auc"] == 1.0


def test_score_on_all_wrong_predictions_is_zero_for_precision_recall():
    worst = score([1, 1], [0, 0], [0.1, 0.2])
    assert worst["precision"] == 0.0
    assert worst["recall"] == 0.0
