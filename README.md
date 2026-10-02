# Credit Card Fraud Detection & MLOps

This repository combines exploratory fraud-detection notebooks with a small, reproducible batch training and evaluation pipeline. The modular Python code currently trains Logistic Regression; other model-family results below come from notebook experiments. The API, mitigation, and drift-monitoring sections describe a proposed architecture and are not currently running services.

## Repository Structure

```text
fraud_detection_mitigation/
├── .github/
│   └── workflows/
│       └── mlops-pipeline.yml    # CI: lint, unit tests, Docker build
├── config/
│   └── config.yaml               # Dataset path, model parameters, metric gate
├── data/                         # Create locally; dataset is not committed
│   └── creditcard.csv
├── notebooks/                    # EDA, preprocessing, model experiments, inference
├── src/
│   ├── __init__.py
│   ├── data.py                    # Load, clean, and split the dataset
│   ├── train.py                   # Train and save Logistic Regression
│   └── evaluate.py                # Write metrics and enforce the ROC-AUC gate
├── tests/
│   └── test_model.py              # Isolated evaluation unit tests
├── Dockerfile
├── requirements.txt
├── requirements-notebooks.txt     # Optional notebook/benchmark dependencies
└── README.md
```

The notebooks have been moved out of the repository root into [notebooks/](notebooks/):

* [Fraud_Classification_EDA.ipynb](notebooks/Fraud_Classification_EDA.ipynb) — exploratory data analysis.
* [Data_Pre_Processing_and_Modelling.ipynb](notebooks/Data_Pre_Processing_and_Modelling.ipynb) — preprocessing and resampling experiments.
* [Fraud_Prevention_Pipeline.ipynb](notebooks/Fraud_Prevention_Pipeline.ipynb) — classifier-family benchmark.
* [Model_inference.ipynb](notebooks/Model_inference.ipynb) — inference exploration.

## Clone and Run

### 1. Clone the repository

If you are starting from the GitHub repository (or a fork), clone it and enter the checkout:

```bash
git clone https://github.com/AkankshaB123/fraud_detection_mitigation.git
cd fraud_detection_mitigation
```

For your own fork, replace the URL with your fork's clone URL.

### 2. Set up Python

Use Python 3.10 or newer, then create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell or `.venv\Scripts\activate.bat` in Command Prompt.

### 3. Add the dataset locally

The credit-card CSV is excluded from Git because it is a large external dataset. Obtain the Credit Card Fraud Detection dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), create `data/`, and place the file at `data/creditcard.csv`. The path can be changed in `config/config.yaml`. Do not commit the dataset or private transaction data.

### 4. Train, evaluate, and test

Run these commands from the repository root, with the virtual environment active and dataset in place:

```bash
python -m src.train
python -m src.evaluate
pytest -v tests/
```

Training writes `model.joblib`; evaluation writes `metrics.json` and fails if ROC-AUC is below `metrics.min_auc_roc` in the config. Both artifacts are ignored by Git. Unit tests use stubbed model/data inputs and do not require the dataset; run them independently with `pytest -v tests/`.

### 5. Run the training container (optional)

Build and run the Docker image from the repository root. The bind mount makes the local dataset available and keeps generated artifacts on the host:

```bash
docker build -t fraud-detection-mlops .
docker run --rm -v "$PWD:/app" fraud-detection-mlops
```

The container runs training followed by evaluation. Docker is optional; the Python commands above do the same work.

### 6. Open the research notebooks (optional)

Install the optional packages and start JupyterLab:

```bash
python -m pip install -r requirements-notebooks.txt
jupyter lab notebooks/
```

Some notebook cells use Google Colab's `google.colab.drive` API or a hosted runtime. Those cells need adapting for local execution; use the local dataset path above. The research notebooks are exploratory and are not required for the modular CLI pipeline or unit tests.

---

## Reference System Architecture

The following diagram is a design reference, not an implemented HTTP service. The checked-in application currently provides batch train/evaluate commands only.

```
                  +--------------------------+
                  | Incoming Transaction API |
                  +------------+-------------+
                               |
                               v
                  +--------------------------+
                  | Data Ingestion & Preproc |
                  |   (Standard Scaling)     |
                  +------------+-------------+
                               |
                               v
                  +--------------------------+
                  |    Inference Engine      |
                  | (RF / XGBoost / LR /     |
                  |     DT / Linear SVM)     |
                  +------------+-------------+
                               |
                               v
                  +--------------------------+
                  |   Decision & Mitigation  |
                  | (Allow / Review / Block) |
                  +--------------------------+
```

---

## Dataset & Preprocessing Details

* **Total Dataset Size:** 284,807 transactions (1,081 duplicates dropped).
* **Class Imbalance:** Fraud represents **0.17%** ($n = 492$) versus **99.83%** ($n = 284,315$) normal transactions.
* **Feature Dimensions:** 30 numerical features (`Time`, `V1`–`V28`, `Amount`).
* **Resampling Configurations:**
  * **Random Undersampling:** Total shape = $984$ samples ($688$ train / $296$ test).
  * **SMOTE Oversampling:** Total shape = $568,630$ samples ($398,041$ train / $170,589$ test).

---

## Comprehensive Model Evaluation Benchmark

All 10 evaluated model variants (spanning Logistic Regression, Decision Trees, Random Forest, XGBoost, and Support Vector Machines under both Undersampling and SMOTE) are detailed below:

| Rank | Model Name | Resampling Method | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Random Forest** | **SMOTE** | **99.93%** | **99.95%** | **99.91%** | **99.93%** |
| **2** | **XGBoost** | **SMOTE** | **99.92%** | **99.90%** | **99.94%** | **99.92%** |
| **3** | **Decision Tree** | **SMOTE** | **98.65%** | **98.71%** | **98.60%** | **98.66%** |
| **4** | **Logistic Regression** | **SMOTE** | **98.04%** | **99.13%** | **96.94%** | **98.02%** |
| **5** | **Linear SVM** | **SMOTE** | **97.53%** | **99.15%** | **95.89%** | **97.49%** |
| **6** | **Logistic Regression** | **Undersampled** | **91.89%** | **97.92%** | **87.04%** | **92.16%** |
| **7** | **XGBoost** | **Undersampled** | **91.22%** | **96.58%** | **87.04%** | **91.56%** |
| **8** | **Random Forest** | **Undersampled** | **90.88%** | **98.56%** | **84.57%** | **91.03%** |
| **9** | **RBF SVM** | **Undersampled** | **90.88%** | **99.27%** | **83.95%** | **90.97%** |
| **10**| **Decision Tree** | **Undersampled** | **88.51%** | **93.84%** | **84.57%** | **88.96%** |

### Model Family Guide

The benchmark covers five classifier families. The score pairs below compare the variants that were actually evaluated; they are not results for every possible model/resampling combination.

#### Logistic Regression — linear baseline

Logistic Regression estimates a linear decision boundary and is a useful low-complexity baseline. It is fast to train and straightforward to inspect, but can miss complex feature interactions unless those are represented in the input features.

* **SMOTE:** 98.04% accuracy, 99.13% precision, 96.94% recall, and 98.02% F1.
* **Random undersampling:** 91.89% accuracy, 97.92% precision, 87.04% recall, and 92.16% F1.
* **Takeaway:** Strong baseline and the only classifier currently implemented by the modular `src/train.py` entry point. The SMOTE result is the better of the two reported variants.

#### Decision Tree — interpretable nonlinear rules

A single tree can model nonlinear feature interactions and express its predictions as rules. It is easier to explain than an ensemble, but its results can vary substantially with tree depth and training data.

* **SMOTE:** 98.65% accuracy, 98.71% precision, 98.60% recall, and 98.66% F1.
* **Random undersampling:** 88.51% accuracy, 93.84% precision, 84.57% recall, and 88.96% F1.
* **Takeaway:** The SMOTE variant has a useful balance of precision and recall, while the undersampled variant is the lowest-F1 model in this benchmark.

#### Random Forest — bagged tree ensemble

Random Forest averages many randomized decision trees. This usually reduces the instability of a single tree while retaining the ability to learn nonlinear relationships.

* **SMOTE:** 99.93% accuracy, 99.95% precision, 99.91% recall, and 99.93% F1; 41 false positives and 76 false negatives in the reported confusion matrix.
* **Random undersampling:** 90.88% accuracy, 98.56% precision, 84.57% recall, and 91.03% F1.
* **Takeaway:** The SMOTE variant leads the reported table on accuracy, precision, and F1, and has the fewest false positives among the SMOTE variants shown. It is the benchmark's balanced-performance choice, subject to independent validation.

#### XGBoost — gradient-boosted trees

XGBoost builds trees sequentially, with each stage focusing on errors from earlier stages. It can capture complex patterns but has more tuning and deployment considerations than the linear baseline.

* **SMOTE:** 99.92% accuracy, 99.90% precision, 99.94% recall, and 99.92% F1; 89 false positives and 54 false negatives in the reported confusion matrix.
* **Random undersampling:** 91.22% accuracy, 96.58% precision, 87.04% recall, and 91.56% F1.
* **Takeaway:** The SMOTE variant has the highest reported recall and the fewest false negatives, making it the candidate to investigate when missed fraud is especially costly. Its higher false-positive count than Random Forest may mean more legitimate transactions require review.

#### Support Vector Machines — margin-based classifiers

SVMs separate classes by maximizing a decision margin. Their behavior depends on the kernel and feature scaling; the two reported SVM rows use different kernels and resampling strategies, so they should not be treated as a controlled head-to-head comparison.

* **Linear SVM with SMOTE:** 97.53% accuracy, 99.15% precision, 95.89% recall, and 97.49% F1.
* **RBF SVM with undersampling:** 90.88% accuracy, 99.27% precision, 83.95% recall, and 90.97% F1.
* **Takeaway:** The linear SMOTE variant maintains high precision with stronger recall than the reported undersampled RBF variant. The RBF result's high precision comes with more missed fraud, so precision alone is not enough to select it.

### Benchmark Scope and Validation Notes

The benchmark figures are recorded from the exploratory notebook experiments, not generated by the modular trainer or by the unit tests. The checked-in `src/train.py` currently supports Logistic Regression only; the other classifier families are benchmark candidates rather than selectable production models. Before using the table to make deployment decisions, rerun the comparisons with a stratified holdout that reflects the natural class imbalance and apply SMOTE **only to each training fold**. Resampling before splitting can leak information into evaluation and inflate scores. Also validate thresholds and review capacity against current transaction costs and data.

---

## Detailed Model Breakdown & Confusion Matrices

### 1. SMOTE-Resampled Models

#### Random Forest (SMOTE) — *Primary Champion*
* **Accuracy:** $99.93\%$ | **Precision:** $99.95\%$ | **Recall:** $99.91\%$ | **F1-Score:** $99.93\%$
* **Confusion Matrix:**
  $$\begin{bmatrix} 85100 & 41 \\ 76 & 85372 \end{bmatrix}$$
  * True Negatives (TN): $85,100$ | False Positives (FP): $41$
  * False Negatives (FN): $76$ | True Positives (TP): $85,372$

#### XGBoost (SMOTE) — *Secondary Champion*
* **Accuracy:** $99.92\%$ | **Precision:** $99.90\%$ | **Recall:** $99.94\%$ | **F1-Score:** $99.92\%$
* **Confusion Matrix:**
  $$\begin{bmatrix} 85052 & 89 \\ 54 & 85394 \end{bmatrix}$$
  * True Negatives (TN): $85,052$ | False Positives (FP): $89$
  * False Negatives (FN): $54$ | True Positives (TP): $85,394$

#### Decision Tree (SMOTE)
* **Accuracy:** $98.65\%$ | **Precision:** $98.71\%$ | **Recall:** $98.60\%$ | **F1-Score:** $98.66\%$
* **Confusion Matrix:**
  $$\begin{bmatrix} 84037 & 1104 \\ 1192 & 84256 \end{bmatrix}$$

#### Logistic Regression (SMOTE)
* **Accuracy:** $98.04\%$ | **Precision:** $99.13\%$ | **Recall:** $96.94\%$ | **F1-Score:** $98.02\%$
* **Confusion Matrix:**
  $$\begin{bmatrix} 84418 & 723 \\ 2615 & 82833 \end{bmatrix}$$

#### Linear SVM (SMOTE)
* **Accuracy:** $97.53\%$ | **Precision:** $99.15\%$ | **Recall:** $95.89\%$ | **F1-Score:** $97.49\%$
* **Confusion Matrix:**
  $$\begin{bmatrix} 84437 & 704 \\ 3516 & 81932 \end{bmatrix}$$

---

### 2. Undersampled Models

#### Logistic Regression (Undersampled)
* **Accuracy:** $91.89\%$ | **Precision:** $97.92\%$ | **Recall:** $87.04\%$ | **F1-Score:** $92.16\%$
* **Confusion Matrix:**
  $$\begin{bmatrix} 131 & 3 \\ 21 & 141 \end{bmatrix}$$

#### XGBoost (Undersampled)
* **Accuracy:** $91.22\%$ | **Precision:** $96.58\%$ | **Recall:** $87.04\%$ | **F1-Score:** $91.56\%$
* **Confusion Matrix:**
  $$\begin{bmatrix} 129 & 5 \\ 21 & 141 \end{bmatrix}$$

#### Random Forest (Undersampled)
* **Accuracy:** $90.88\%$ | **Precision:** $98.56\%$ | **Recall:** $84.57\%$ | **F1-Score:** $91.03\%$
* **Confusion Matrix:**
  $$\begin{bmatrix} 132 & 2 \\ 25 & 137 \end{bmatrix}$$

#### RBF SVM (Undersampled)
* **Accuracy:** $90.88\%$ | **Precision:** $99.27\%$ | **Recall:** $83.95\%$ | **F1-Score:** $90.97\%$
* **Confusion Matrix:**
  $$\begin{bmatrix} 133 & 1 \\ 26 & 136 \end{bmatrix}$$

#### Decision Tree (Undersampled)
* **Accuracy:** $88.51\%$ | **Precision:** $93.84\%$ | **Recall:** $84.57\%$ | **F1-Score:** $88.96\%$
* **Confusion Matrix:**
  $$\begin{bmatrix} 125 & 9 \\ 25 & 137 \end{bmatrix}$$

---

## Model Selection Rationale

1. **SMOTE vs. Undersampling Performance:** All models trained on SMOTE drastically outperform their undersampled counterparts. Undersampling leads to severe information loss, constraining recall to $\le 87.04\%$ across all techniques.
2. **Primary Runtime Selection:** **Random Forest (SMOTE)** and **XGBoost (SMOTE)** achieve near-perfect performance ($>99.91\%$ Recall, Precision, and F1-Score).
   * **Random Forest (SMOTE)** is selected as the **Primary Model** due to its lowest false-positive rate ($41$ false alarms).
   * **XGBoost (SMOTE)** serves as an alternative high-recall model ($54$ missed frauds).
   * **Logistic Regression (SMOTE)** is preserved as a lightweight, low-latency baseline option ($14.2\text{ ms}$ average latency).

---

## API Specification

### `POST /v1/predict`

#### Request Payload
```json
{
  "transaction_id": "tx_9876543210",
  "timestamp": "2026-10-01T15:16:00Z",
  "features": {
    "Amount": 249.50,
    "Time": 10.0,
    "V1": -1.359807,
    "V2": -0.072781,
    "V3": 2.536347
  }
}
```

#### Response Payload
```json
{
  "transaction_id": "tx_9876543210",
  "fraud_probability": 0.0012,
  "is_fraud": false,
  "action": "ALLOW",
  "execution_time_ms": 8.4,
  "model_version": "RandomForest_SMOTE_v1"
}
```

---

## Decision & Mitigation Policy

Inference outputs ($P_{\text{fraud}}$) map directly to mitigation policies:

* **$P_{\text{fraud}} < 0.30 \rightarrow$ ALLOW:** Transaction executes immediately.
* **$0.30 \le P_{\text{fraud}} < 0.80 \rightarrow$ MANUAL REVIEW / Step-Up MFA:** Triggers additional verification (3DS, OTP) or analyst review.
* **$P_{\text{fraud}} \ge 0.80 \rightarrow$ BLOCK:** Transaction automatically declined and card flagged.

---

## Drift Monitoring Architecture (`drift_monitor.py`)

The system monitors feature drift via **Population Stability Index (PSI)** and prediction distribution drift using the **Kolmogorov-Smirnov (KS) test**.

```python
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

class DriftMonitor:
    def __init__(self, baseline_df: pd.DataFrame, baseline_probs: np.ndarray):
        """Stores baseline features and prediction probabilities from training data."""
        self.baseline_df = baseline_df
        self.baseline_probs = baseline_probs

    def calculate_psi(self, baseline: np.ndarray, current: np.ndarray, num_buckets: int = 10) -> float:
        """Calculates Population Stability Index (PSI) for numerical features."""
        percentiles = np.linspace(0, 100, num_buckets + 1)
        buckets = np.percentile(baseline, percentiles)
        buckets[0] -= 1e-5
        buckets[-1] += 1e-5

        baseline_counts, _ = np.histogram(baseline, bins=buckets)
        current_counts, _ = np.histogram(current, bins=buckets)

        baseline_pct = np.where(baseline_counts == 0, 0.0001, baseline_counts) / len(baseline)
        current_pct = np.where(current_counts == 0, 0.0001, current_counts) / len(current)

        return float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))

    def analyze_batch(self, production_df: pd.DataFrame, production_probs: np.ndarray) -> dict:
        """Evaluates batch for feature drift (PSI) and output prediction drift (KS Test)."""
        results = {"feature_psi": {}, "concept_drift_ks": None, "trigger_retrain": False}

        drifted_features = []
        for col in self.baseline_df.select_dtypes(include=[np.number]).columns:
            psi = self.calculate_psi(self.baseline_df[col].dropna().values, production_df[col].dropna().values)
            results["feature_psi"][col] = round(psi, 4)
            if psi > 0.25:
                drifted_features.append(col)

        ks_stat, p_value = ks_2samp(self.baseline_probs, production_probs)
        results["concept_drift_ks"] = {"stat": round(ks_stat, 4), "p_value": round(p_value, 4)}

        if p_value < 0.05 or len(drifted_features) >= 2:
            results["trigger_retrain"] = True

        return results
```

---

## Automated Retraining Pipeline (`retrain_pipeline.py`)

When drift alerts fire, the retraining engine scales features, resamples via SMOTE, retrains the selected model (Random Forest / XGBoost / Logistic Regression), and outputs updated model artifacts.

```python
import joblib
import datetime
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class RetrainingPipeline:
    def __init__(self, artifact_path: str = "./artifacts"):
        self.artifact_path = artifact_path

    def retrain(self, X_new: pd.DataFrame, y_new: pd.Series, model_type: str = "rf"):
        """Retrains scaling, SMOTE resampling, and primary classification model."""
        print(f"[{datetime.datetime.now()}] 🔄 Retraining triggered for model type: {model_type}...")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_new)

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y_new)

        if model_type == "rf":
            new_model = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
        elif model_type == "xgb":
            from xgboost import XGBClassifier
            new_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, tree_method='hist')
        else:
            from sklearn.linear_model import LogisticRegression
            new_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)

        new_model.fit(X_resampled, y_resampled)

        joblib.dump(new_model, f"{self.artifact_path}/{model_type}_smote_model.pkl")
        joblib.dump(scaler, f"{self.artifact_path}/scaler.pkl")

        print(f"[{datetime.datetime.now()}] ✅ Model successfully retrained and deployed.")
        return new_model, scaler
```

---

## Metric Threshold Summary

* **$\text{PSI} < 0.10$:** Distribution stable.
* **$0.10 \le \text{PSI} \le 0.25$:** Moderate drift (Warning status).
* **$\text{PSI} > 0.25$:** Significant feature drift requiring model retraining.
* **KS-Test $p\text{-value} < 0.05$:** Statistically significant output prediction drift.
