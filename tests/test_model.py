import json

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from streamlit.testing.v1 import AppTest

from src import evaluate
from src.modeling import (
    DEFAULT_PARAMS,
    MODEL_NAMES,
    evaluate_predictions,
    fit_and_evaluate,
)


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


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_all_model_choices_fit_and_produce_probabilities(model_name):
    features, labels = make_classification(
        n_samples=120,
        n_features=8,
        n_informative=5,
        weights=[0.8, 0.2],
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=42, stratify=labels
    )
    params = DEFAULT_PARAMS[model_name].copy()
    if model_name == "Random Forest":
        params["n_estimators"] = 20
    elif model_name == "XGBoost":
        params["n_estimators"] = 10

    model, metrics = fit_and_evaluate(
        model_name,
        params,
        False,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    probabilities = model.predict_proba(X_test)[:, 1]
    assert probabilities.shape == (len(X_test),)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_smote_is_applied_inside_training_pipeline():
    features, labels = make_classification(
        n_samples=180,
        n_features=8,
        n_informative=5,
        weights=[0.85, 0.15],
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=42, stratify=labels
    )

    model, metrics = fit_and_evaluate(
        "Logistic Regression",
        DEFAULT_PARAMS["Logistic Regression"],
        True,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    assert "smote" in model.named_steps
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["confusion_matrix"].shape == (2, 2)


def test_decision_threshold_changes_binary_predictions():
    labels = np.array([0, 1, 1])
    probabilities = np.array([0.4, 0.55, 0.9])

    permissive = evaluate_predictions(labels, probabilities, threshold=0.5)
    conservative = evaluate_predictions(labels, probabilities, threshold=0.6)

    assert permissive["predictions"].tolist() == [0, 1, 1]
    assert conservative["predictions"].tolist() == [0, 0, 1]


def test_streamlit_app_renders_without_dataset():
    app = AppTest.from_file("streamlit_app.py").run(timeout=30)

    assert not app.exception
    assert [item.value for item in app.title] == ["🛡️ Fraud Detection Model Lab"]