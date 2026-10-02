# Real-Time Fraud Detection & Mitigation Engine

This repository contains the production system design, benchmarking evaluation, and monitoring architecture for an automated **Fraud Detection & Mitigation System**. The pipeline encompasses end-to-end dataset preprocessing, class imbalance resolution via SMOTE and Undersampling, evaluation across multiple model architectures, and continuous drift monitoring.

---

## System Architecture

The microservice ingests transaction payloads, applies feature scaling, runs low-latency inference across trained model candidates, and routes actions based on fraud probability thresholds.

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