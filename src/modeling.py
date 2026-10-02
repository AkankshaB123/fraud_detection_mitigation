"""Model construction and evaluation helpers for the fraud demo app."""

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "XGBoost",
    "Linear SVM",
    "RBF SVM",
]

DEFAULT_PARAMS = {
    "Logistic Regression": {"C": 1.0, "max_iter": 1000, "class_weight": None},
    "Decision Tree": {
        "max_depth": 8,
        "min_samples_leaf": 1,
        "class_weight": None,
    },
    "Random Forest": {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_leaf": 1,
        "class_weight": None,
    },
    "XGBoost": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
    "Linear SVM": {"C": 1.0, "class_weight": None},
    "RBF SVM": {"C": 1.0, "gamma": "scale", "class_weight": None},
}


def create_estimator(model_name, params):
    """Create one supported classifier with the requested hyperparameters."""
    if model_name == "Logistic Regression":
        return LogisticRegression(
            C=params["C"],
            max_iter=params["max_iter"],
            class_weight=params["class_weight"],
            random_state=42,
        )
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight=params["class_weight"],
            random_state=42,
        )
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight=params["class_weight"],
            random_state=42,
            n_jobs=-1,
        )
    if model_name == "XGBoost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "Install optional XGBoost with: pip install xgboost"
            ) from exc
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    if model_name == "Linear SVM":
        return SVC(
            C=params["C"],
            kernel="linear",
            class_weight=params["class_weight"],
            probability=True,
            random_state=42,
        )
    if model_name == "RBF SVM":
        return SVC(
            C=params["C"],
            gamma=params["gamma"],
            class_weight=params["class_weight"],
            probability=True,
            random_state=42,
        )
    raise ValueError("Unsupported model: {}".format(model_name))


def build_model(model_name, params=None, use_smote=False):
    """Build a leakage-safe preprocessing/sampling/classifier pipeline."""
    if model_name not in MODEL_NAMES:
        raise ValueError("Unsupported model: {}".format(model_name))

    params = params or DEFAULT_PARAMS[model_name]
    steps = []
    if model_name in ("Logistic Regression", "Linear SVM", "RBF SVM"):
        steps.append(("scale", StandardScaler()))
    if use_smote:
        steps.append(("smote", SMOTE(random_state=42)))
    steps.append(("classifier", create_estimator(model_name, params)))

    if use_smote:
        return Pipeline(steps)
    return SklearnPipeline(steps)


def evaluate_predictions(y_true, probabilities, threshold=0.5):
    """Return threshold-based metrics and ROC-AUC for a held-out test set."""
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "confusion_matrix": matrix,
        "predictions": predictions,
    }


def fit_and_evaluate(
    model_name,
    params,
    use_smote,
    X_train,
    X_test,
    y_train,
    y_test,
    threshold=0.5,
):
    """Fit on training data only and score untouched held-out data."""
    model = build_model(model_name, params=params, use_smote=use_smote)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_predictions(y_test, probabilities, threshold=threshold)
    return model, metrics
