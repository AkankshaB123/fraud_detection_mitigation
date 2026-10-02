mkdir -p fraud-detection-mlops/.github/workflows \
         fraud-detection-mlops/config \
         fraud-detection-mlops/src \
         fraud-detection-mlops/tests && \
cd fraud-detection-mlops && \
cat << 'EOF' > README.md
# Credit Card Fraud Detection MLOps Pipeline

Production-ready MLOps pipeline for Credit Card Fraud Detection, converted from exploratory notebooks into a modular, containerized, and automated CI/CD workflow.

## Project Structure

```text
fraud-detection-mlops/
├── .github/
│   └── workflows/
│       └── mlops-pipeline.yml
├── config/
│   └── config.yaml
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── train.py
│   └── evaluate.py
├── tests/
│   └── test_model.py
├── Dockerfile
├── requirements.txt
└── README.md