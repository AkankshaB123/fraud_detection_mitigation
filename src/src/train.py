import joblib
import yaml
from sklearn.linear_model import LogisticRegression
from src.data import load_and_clean_data


def train_model():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    X_train, X_test, y_train, y_test = load_and_clean_data()

    if config["model"]["type"] == "LogisticRegression":
        model = LogisticRegression(**config["model"]["params"])
    else:
        raise ValueError(f"Unsupported model type: {config['model']['type']}")

    model.fit(X_train, y_train)

    # Save model artifact
    joblib.dump(model, "model.joblib")
    return model, X_test, y_test


if __name__ == "__main__":
    train_model()