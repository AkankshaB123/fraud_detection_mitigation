import json
import os


def test_metrics_exist():
    assert os.path.exists("metrics.json"), "Metrics file was not generated."


def test_model_performance():
    with open("metrics.json", "r") as f:
        metrics = json.load(f)

    assert metrics["roc_auc"] >= 0.80, f"ROC-AUC score {metrics['roc_auc']} is too low."
    assert metrics["f1_score"] > 0.0, "F1 score must be greater than zero."