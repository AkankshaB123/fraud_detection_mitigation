# Real-Time Fraud Detection & Mitigation Engine

This repository contains the production system design and monitoring architecture for an automated **Fraud Detection & Mitigation System** powered by a Logistic Regression model trained with SMOTE resampling.

---

## System Architecture

The microservice ingests transaction payloads, applies feature encoding and scaling, runs low-latency inference using the trained **Logistic Regression** model, and applies mitigation rules based on fraud probability thresholds.

```
                  +--------------------------+
                  | Incoming Transaction API |
                  +------------+-------------+
                               |
                               v
                  +--------------------------+
                  | Data Ingestion & Preproc |
                  |  (Encoding & Scaling)    |
                  +------------+-------------+
                               |
                               v
                  +--------------------------+
                  | Logistic Regression Engine|
                  |  (SMOTE-Trained Model)   |
                  +------------+-------------+
                               |
                               v
                  +--------------------------+
                  |   Decision & Mitigation  |
                  | (Allow / Review / Block) |
                  +--------------------------+
```

---

## Model Evaluation & Resampling Results

To resolve extreme class imbalance, both **Random Undersampling** and **SMOTE Oversampling** strategies were evaluated using Logistic Regression models:

| Metric | Logistic Regression (Undersampled) | Logistic Regression (SMOTE - Selected) |
| :--- | :--- | :--- |
| **Accuracy** | 64.53% | **98.04%** |
| **Precision** | 80.00% | **99.13%** |
| **Recall** | 46.91% | **96.94%** |
| **F1-Score** | 59.14% | **98.02%** |
| **True Negatives (TN)** | 115 | **84,418** |
| **False Positives (FP)** | 19 | **723** |
| **False Negatives (FN)** | 86 | **2,615** |
| **True Positives (TP)** | 76 | **82,833** |

### Rationale
The SMOTE-trained Logistic Regression model was selected because the undersampled model produced high False Negatives ($FN = 86$ out of $162$ fraud cases, yielding $46.91\%$ recall). SMOTE achieves **96.94% Recall** and **99.13% Precision**, substantially limiting financial loss from missed fraudulent activity.

---

## API Specification

### `POST /v1/predict`

#### Request Payload
```json
{
  "transaction_id": "tx_9876543210",
  "timestamp": "2026-10-01T15:16:00Z",
  "features": {
    "amount": 249.50,
    "location_code": "US-CA",
    "device_risk_score": 0.12,
    "velocity_1h": 2
  }
}
```

#### Response Payload
```json
{
  "transaction_id": "tx_9876543210",
  "fraud_probability": 0.0124,
  "is_fraud": false,
  "action": "ALLOW",
  "execution_time_ms": 14.2
}
```

---

## Decision & Mitigation Policy

Inference outputs ($P_{fraud}$) map directly to decision rules:

* **$P_{fraud} < 0.30 \rightarrow$ ALLOW:** Transaction executes automatically.
* **$0.30 \le P_{fraud} < 0.85 \rightarrow$ MANUAL REVIEW / Step-Up MFA:** Triggers additional authentication (3DS, OTP) or manual analyst review.
* **$P_{fraud} \ge 0.85 \rightarrow$ BLOCK:** Transaction automatically declined; user account flagged for security audit.

---

## Drift Monitoring & Retraining Architecture

### 1. Drift Monitor Class (`drift_monitor.py`)

The system monitors input feature drift using **Population Stability Index (PSI)** and prediction drift using the **Kolmogorov-Smirnov (KS) test**.

```python
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

class DriftMonitor:
    def __init__(self, baseline_df: pd.DataFrame, baseline_probs: np.ndarray):
        """Stores baseline features and predictions from the training set."""
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
        """Evaluates batch for feature drift (PSI) and prediction drift (KS Test)."""
        results = {"feature_psi": {}, "concept_drift_ks": None, "trigger_retrain": False}
        
        # Feature Drift (PSI)
        drifted_features = []
        for col in self.baseline_df.select_dtypes(include=[np.number]).columns:
            psi = self.calculate_psi(self.baseline_df[col].dropna().values, production_df[col].dropna().values)
            results["feature_psi"][col] = round(psi, 4)
            if psi > 0.25:  # Threshold for significant drift
                drifted_features.append(col)

        # Output Prediction Drift (KS-Test)
        ks_stat, p_value = ks_2samp(self.baseline_probs, production_probs)
        results["concept_drift_ks"] = {"stat": round(ks_stat, 4), "p_value": round(p_value, 4)}

        # Trigger condition: Prediction distribution shift OR >=2 feature drifts
        if p_value < 0.05 or len(drifted_features) >= 2:
            results["trigger_retrain"] = True

        return results
```

### 2. Automated Retraining Pipeline (`retrain_pipeline.py`)

When drift is detected, the pipeline re-encodes, rescales, resamples using SMOTE, retrains the Logistic Regression model, and updates serialized disk artifacts.

```python
import joblib
import datetime
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

class RetrainingPipeline:
    def __init__(self, artifact_path: str = "./artifacts"):
        self.artifact_path = artifact_path

    def retrain(self, X_new: pd.DataFrame, y_new: pd.Series):
        """Retrains preprocessing transformers and Logistic Regression model."""
        print(f"[{datetime.datetime.now()}] 🔄 Retraining triggered...")

        # Preprocess features
        X_scaled, encoder, scaler = preprocess_features(X_new, is_training=True)

        # SMOTE Oversampling
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y_new)

        # Retrain Logistic Regression model
        new_model = LogisticRegression(max_iter=1000, random_state=42)
        new_model.fit(X_resampled, y_resampled)

        # Save artifacts
        joblib.dump(new_model, f"{self.artifact_path}/model_smote.pkl")
        joblib.dump(scaler, f"{self.artifact_path}/scaler_over.pkl")
        joblib.dump(encoder, f"{self.artifact_path}/encoder_over.pkl")

        print(f"[{datetime.datetime.now()}] ✅ Model successfully retrained.")
        return new_model, scaler, encoder
```

---

## Metric Threshold Summary

* **PSI $< 0.10$:** Stable feature distribution.
* **$0.10 \le \text{PSI} \le 0.25$:** Slight drift (Warning state).
* **PSI $> 0.25$:** Significant drift requiring model update.
* **KS-Test $p\text{-value} < 0.05$:** Statistically significant output drift.
