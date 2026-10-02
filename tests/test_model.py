import json

import numpy as np
import pytest

from src import evaluate


class StubModel:
    def predict(self, features):
        return [0, 1, 1, 1]

    def predict_proba(self, features):
        return np.array([[0.9, 0.1], [0.3, 0.7], [0.4, 0.6], [0.1, 0.9]])


def configure_evaluation(tmp_path, monkeypatch, min_auc_roc=0.7):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        f"metrics:\n  min_auc_roc: {min_auc_roc}\n"
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(evaluate.joblib, "load", lambda _: StubModel())
    monkeypatch.setattr(
        evaluate,
        "load_and_clean_data",
        lambda: (None, None, None, [0, 0, 1, 1]),
    )


def test_evaluation_writes_metrics(tmp_path, monkeypatch):
    configure_evaluation(tmp_path, monkeypatch)

    evaluate.evaluate_model()

    metrics_path = tmp_path / "metrics.json"
    assert metrics_path.exists(), "Metrics file was not generated."
    metrics = json.loads(metrics_path.read_text())
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["f1_score"] == pytest.approx(0.8)


def test_evaluation_rejects_auc_below_threshold(tmp_path, monkeypatch):
    configure_evaluation(tmp_path, monkeypatch, min_auc_roc=0.8)

    with pytest.raises(ValueError, match="below threshold"):
        evaluate.evaluate_model()