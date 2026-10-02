import json
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
import yaml
from src.data import load_and_clean_data


def evaluate_model():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model = joblib.load("model.joblib")
    _, X_test, _, y_test = load_and_clean_data()

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Performance Gate
    if metrics["roc_auc"] < config["metrics"]["min_auc_roc"]:
        raise ValueError(
            f"Model ROC-AUC ({metrics['roc_auc']:.4f}) is below threshold ({config['metrics']['min_auc_roc']})!"
        )

    print("Model evaluation passed successfully.")


if __name__ == "__main__":
    evaluate_model()