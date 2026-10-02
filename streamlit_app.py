"""Interactive fraud-model training, comparison, and prediction demo."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from src.modeling import (
    DEFAULT_PARAMS,
    MODEL_NAMES,
    evaluate_predictions,
    fit_and_evaluate,
)

st.set_page_config(
    page_title="Fraud Detection Model Lab",
    page_icon="🛡️",
    layout="wide",
)

FEATURE_NAMES = ["Time"] + ["V{}".format(index) for index in range(1, 29)] + ["Amount"]
DATA_PATH = Path("data/creditcard.csv")

st.title("🛡️ Fraud Detection Model Lab")
st.caption(
    "Train and compare classifiers on a held-out test set, tune model "
    "hyperparameters, and score transaction feature rows."
)
st.warning(
    "Research/demo only—not a payment decision system. Model scores are not "
    "guaranteed to be calibrated or accurate in production. Never upload "
    "customer-identifiable or confidential transaction data."
)

with st.sidebar:
    st.header("Data and evaluation")
    uploaded_dataset = st.file_uploader(
        "Credit-card CSV (must include Class)", type=["csv"], key="dataset"
    )
    st.caption("Expected target: Class (0 = legitimate, 1 = fraud).")
    test_fraction = st.slider(
        "Held-out test fraction", min_value=0.1, max_value=0.4, value=0.2, step=0.05
    )
    max_training_rows = st.slider(
        "Maximum training rows",
        min_value=5000,
        max_value=20000,
        value=5000,
        step=2500,
        help="Stratified training cap keeps kernel SVM experiments practical.",
    )
    max_evaluation_rows = st.slider(
        "Maximum held-out evaluation rows",
        min_value=2000,
        max_value=20000,
        value=10000,
        step=2000,
        help="A stratified holdout sample limits prediction cost for kernel SVMs.",
    )
    threshold = st.slider(
        "Fraud decision threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.01,
        help="Predict fraud when the model's estimated fraud probability meets this threshold.",
    )
    use_smote = st.checkbox(
        "Apply SMOTE to training split only",
        value=False,
        help="Synthetic oversampling is confined to the training split; test data stays untouched.",
    )

try:
    if uploaded_dataset is not None:
        raw_df = pd.read_csv(uploaded_dataset)
        data_source = "uploaded CSV"
    elif DATA_PATH.exists():
        raw_df = pd.read_csv(DATA_PATH)
        data_source = str(DATA_PATH)
    else:
        raw_df = None
        data_source = None
except Exception as exc:
    raw_df = None
    data_source = None
    st.error("Could not read the selected CSV: {}".format(exc))

if raw_df is None:
    st.info(
        "Upload the Kaggle credit-card fraud CSV in the sidebar, or put it at "
        "`data/creditcard.csv`, to enable training. The dataset is not included in this repository."
    )
else:
    missing = sorted(set(FEATURE_NAMES + ["Class"]) - set(raw_df.columns))
    if missing:
        st.error("Dataset is missing required columns: {}".format(", ".join(missing)))
    else:
        data = raw_df[FEATURE_NAMES + ["Class"]].copy()
        data = data.apply(pd.to_numeric, errors="coerce").dropna()
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        data = data.drop_duplicates()
        data = data[data["Class"].isin([0, 1])]
        data["Class"] = data["Class"].astype(int)

        if data["Class"].nunique() != 2 or data["Class"].value_counts().min() < 2:
            st.error("The CSV needs at least two examples of each Class (0 and 1).")
        else:
            experiment_key = hashlib.sha256(
                pd.util.hash_pandas_object(data, index=True).values.tobytes()
                + str(
                    (
                        test_fraction,
                        use_smote,
                        max_training_rows,
                        max_evaluation_rows,
                    )
                ).encode("utf-8")
            ).hexdigest()
            if st.session_state.get("experiment_key") != experiment_key:
                for stale_key in (
                    "fitted_model",
                    "trained_model_name",
                    "trained_features",
                    "training_medians",
                    "last_metrics",
                    "last_test_labels",
                    "last_test_probabilities",
                    "comparison_results",
                ):
                    st.session_state.pop(stale_key, None)
                st.session_state["experiment_key"] = experiment_key

            st.caption(
                "Data source: {} · {:,} usable rows after numeric, duplicate, and target checks".format(
                    data_source, len(data)
                )
            )
            total_rows = len(data)
            fraud_count = int(data["Class"].sum())
            summary = st.columns(3)
            summary[0].metric("Transactions", "{:,}".format(total_rows))
            summary[1].metric("Fraud cases", "{:,}".format(fraud_count))
            summary[2].metric(
                "Fraud prevalence", "{:.3f}%".format(100 * fraud_count / total_rows)
            )

            X = data[FEATURE_NAMES]
            y = data["Class"]
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_fraction,
                random_state=42,
                stratify=y,
            )
            if len(X_train) > max_training_rows:
                X_train, _, y_train, _ = train_test_split(
                    X_train,
                    y_train,
                    train_size=max_training_rows,
                    random_state=42,
                    stratify=y_train,
                )
            if len(X_test) > max_evaluation_rows:
                X_test, _, y_test, _ = train_test_split(
                    X_test,
                    y_test,
                    train_size=max_evaluation_rows,
                    random_state=43,
                    stratify=y_test,
                )
            st.caption(
                "Experiment sample: {:,} training rows · {:,} held-out rows. "
                "Both subsets preserve class proportions.".format(
                    len(X_train), len(X_test)
                )
            )

            st.header("Train and compare models")
            model_name = st.selectbox("Model to tune", MODEL_NAMES)
            defaults = DEFAULT_PARAMS[model_name]
            with st.expander("Tune {} hyperparameters".format(model_name), expanded=True):
                if model_name == "Logistic Regression":
                    params = {
                        "C": st.number_input("C (inverse regularization)", 0.001, 100.0, float(defaults["C"]), step=0.1),
                        "max_iter": st.number_input("Maximum iterations", 100, 10000, int(defaults["max_iter"]), step=100),
                        "class_weight": st.selectbox("Class weight", [None, "balanced"], format_func=lambda item: "None" if item is None else item),
                    }
                elif model_name == "Decision Tree":
                    params = {
                        "max_depth": st.number_input("Maximum depth (0 = unlimited)", 0, 100, int(defaults["max_depth"])),
                        "min_samples_leaf": st.number_input("Minimum samples per leaf", 1, 100, int(defaults["min_samples_leaf"])),
                        "class_weight": st.selectbox("Class weight", [None, "balanced"], format_func=lambda item: "None" if item is None else item),
                    }
                    params["max_depth"] = params["max_depth"] or None
                elif model_name == "Random Forest":
                    params = {
                        "n_estimators": st.number_input("Number of trees", 50, 1000, int(defaults["n_estimators"]), step=50),
                        "max_depth": st.number_input("Maximum depth (0 = unlimited)", 0, 100, int(defaults["max_depth"])),
                        "min_samples_leaf": st.number_input("Minimum samples per leaf", 1, 100, int(defaults["min_samples_leaf"])),
                        "class_weight": st.selectbox("Class weight", [None, "balanced"], format_func=lambda item: "None" if item is None else item),
                    }
                    params["max_depth"] = params["max_depth"] or None
                elif model_name == "XGBoost":
                    params = {
                        "n_estimators": st.number_input("Boosting rounds", 50, 1000, int(defaults["n_estimators"]), step=50),
                        "max_depth": st.number_input("Tree depth", 2, 20, int(defaults["max_depth"])),
                        "learning_rate": st.number_input("Learning rate", 0.01, 1.0, float(defaults["learning_rate"]), step=0.01, format="%.2f"),
                    }
                elif model_name == "Linear SVM":
                    params = {
                        "C": st.number_input("C (margin regularization)", 0.001, 100.0, float(defaults["C"]), step=0.1),
                        "class_weight": st.selectbox("Class weight", [None, "balanced"], format_func=lambda item: "None" if item is None else item),
                    }
                else:
                    params = {
                        "C": st.number_input("C (margin regularization)", 0.001, 100.0, float(defaults["C"]), step=0.1),
                        "gamma": st.selectbox("Kernel coefficient", ["scale", "auto"]),
                        "class_weight": st.selectbox("Class weight", [None, "balanced"], format_func=lambda item: "None" if item is None else item),
                    }

            train_col, compare_col = st.columns(2)
            with train_col:
                train_clicked = st.button("Train selected model", type="primary", use_container_width=True)
            with compare_col:
                compare_clicked = st.button("Compare all model defaults", use_container_width=True)

            if train_clicked:
                with st.spinner("Training {}…".format(model_name)):
                    try:
                        fitted, metrics = fit_and_evaluate(
                            model_name,
                            params,
                            use_smote,
                            X_train,
                            X_test,
                            y_train,
                            y_test,
                            threshold,
                        )
                        st.session_state["fitted_model"] = fitted
                        st.session_state["trained_model_name"] = model_name
                        st.session_state["trained_features"] = FEATURE_NAMES
                        st.session_state["training_medians"] = X_train.median()
                        st.session_state["last_test_labels"] = y_test.to_numpy()
                        st.session_state["last_test_probabilities"] = fitted.predict_proba(X_test)[:, 1]
                        st.session_state["last_metrics"] = metrics
                        st.success("{} trained and evaluated.".format(model_name))
                    except Exception as exc:
                        st.error("Training failed: {}".format(exc))

            if compare_clicked:
                rows = []
                failures = []
                progress = st.progress(0.0, text="Comparing models…")
                for index, candidate in enumerate(MODEL_NAMES):
                    try:
                        _, score = fit_and_evaluate(
                            candidate,
                            DEFAULT_PARAMS[candidate],
                            use_smote,
                            X_train,
                            X_test,
                            y_train,
                            y_test,
                            threshold,
                        )
                        rows.append(
                            {
                                "Model": candidate,
                                "Accuracy": score["accuracy"],
                                "Precision": score["precision"],
                                "Recall": score["recall"],
                                "F1": score["f1"],
                                "ROC-AUC": score["roc_auc"],
                            }
                        )
                    except Exception as exc:
                        failures.append("{}: {}".format(candidate, exc))
                    progress.progress((index + 1) / len(MODEL_NAMES), text="Compared {} / {}".format(index + 1, len(MODEL_NAMES)))
                progress.empty()
                if rows:
                    st.session_state["comparison_results"] = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
                for failure in failures:
                    st.warning(failure)

            if "comparison_results" in st.session_state:
                st.subheader("Held-out comparison (sorted by ROC-AUC)")
                st.dataframe(
                    st.session_state["comparison_results"].style.format(
                        {metric: "{:.4f}" for metric in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    "Every candidate uses the same stratified split and training-only preprocessing. "
                    "Comparisons use each model's documented defaults, not your selected model's tuned values."
                )

            if "last_metrics" in st.session_state:
                st.header("Selected model results")
                metrics = evaluate_predictions(
                    st.session_state["last_test_labels"],
                    st.session_state["last_test_probabilities"],
                    threshold,
                )
                st.caption(
                    "{} · fraud threshold {:.2f}".format(
                        st.session_state["trained_model_name"],
                        threshold,
                    )
                )
                metric_cols = st.columns(5)
                for column, label, key in zip(
                    metric_cols,
                    ["Accuracy", "Precision", "Recall", "F1 score", "ROC-AUC"],
                    ["accuracy", "precision", "recall", "f1", "roc_auc"],
                ):
                    column.metric(label, "{:.4f}".format(metrics[key]))
                st.write("Confusion matrix (rows = actual, columns = predicted; class order 0, 1)")
                st.dataframe(
                    pd.DataFrame(
                        metrics["confusion_matrix"],
                        index=["Actual legitimate", "Actual fraud"],
                        columns=["Predicted legitimate", "Predicted fraud"],
                    ),
                    use_container_width=True,
                )
                st.caption(
                    "Accuracy can be misleading on highly imbalanced data. Review recall, precision, "
                    "ROC-AUC, and the false-positive/false-negative counts before choosing a threshold."
                )

            st.header("Score transaction input")
            if "fitted_model" not in st.session_state:
                st.info("Train a model above to enable transaction scoring.")
            else:
                prediction_upload = st.file_uploader(
                    "Optional: upload a CSV with one or more feature rows (no Class column)",
                    type=["csv"],
                    key="prediction_csv",
                )
                if prediction_upload is not None:
                    try:
                        prediction_rows = pd.read_csv(prediction_upload)
                        missing_features = sorted(set(FEATURE_NAMES) - set(prediction_rows.columns))
                        if missing_features:
                            st.error("Prediction CSV is missing: {}".format(", ".join(missing_features)))
                            prediction_rows = None
                        else:
                            prediction_rows = prediction_rows[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
                            if prediction_rows.isna().any().any():
                                st.error("Prediction input must contain numeric values in every feature column.")
                                prediction_rows = None
                    except Exception as exc:
                        st.error("Could not read prediction CSV: {}".format(exc))
                        prediction_rows = None
                else:
                    default_row = st.session_state["training_medians"].to_dict()
                    initial_values = pd.DataFrame([default_row], columns=FEATURE_NAMES)
                    prediction_rows = st.data_editor(
                        initial_values,
                        num_rows="fixed",
                        use_container_width=True,
                        key="manual_prediction_input",
                    )
                    prediction_rows = prediction_rows[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")

                if prediction_rows is not None:
                    if prediction_rows.empty or prediction_rows.isna().any().any():
                        st.error("Provide at least one complete numeric feature row to score.")
                    elif st.button("Score transaction row(s)", key="score_transactions"):
                        active_model = st.session_state["fitted_model"]
                        probability = active_model.predict_proba(prediction_rows)[:, 1]
                        result = prediction_rows.copy()
                        result.insert(0, "Fraud probability", probability)
                        result.insert(1, "Prediction", ["Fraud" if value >= threshold else "Legitimate" for value in probability])
                        st.dataframe(result, use_container_width=True, hide_index=True)
                        st.caption(
                            "Predictions use threshold {:.2f}. Probabilities are estimates, not guarantees.".format(
                                threshold
                            )
                        )
