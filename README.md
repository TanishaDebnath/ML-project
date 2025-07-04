# ML-project

# MLOps End-to-End Example with MLflow

This repository demonstrates a complete MLOps lifecycle for a regression problem using MLflow, with a focus on reproducibility, experiment tracking, and model management.

---

## Project Structure

```
mlproject/
│
├── housing.csv
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── requirements.txt
└── README.md

 ```
## Setup

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Verify that `housing.csv`** is present in the project root.

---

## MLflow Tracking

Start the MLflow UI in your terminal:

```bash
mlflow ui
```

Access the UI at [http://localhost:5000](http://localhost:5000).

---

## Usage

### 1. **Training**

Train the model and log experiments to MLflow:

```bash
python src/train.py --data_path housing.csv
```

This will log parameters, metrics, and the trained model artifact to MLflow.

---

### 2. **Evaluation**

Evaluate any trained model from MLflow:

```bash
python src/evaluate.py --model_uri <model_uri> --data_path housing.csv
```

Replace `<model_uri>` with your MLflow model URI  
(e.g., `runs:/<run_id>/model`).

---

### 3. **Prediction**

To predict with a trained model, provide all feature values in order:

```bash
python src/predict.py --model_uri <model_uri> --features <feature1> <feature2> ... <featureN>
```

Example:
```bash
python src/predict.py --model_uri runs:/<run_id>/model --features -122.23 37.88 41 880 129 322 126 8.3252
```

> Adjust the order and number of features according to your dataset columns.

---

## MLOps Features

- **Experiment tracking:** All metrics and parameters are logged to MLflow.
- **Model registry:** You can promote, version, and deploy models using the MLflow Model Registry (via MLflow UI).
- **Reproducibility:** All code and data are versioned; runs are tracked for full reproducibility.

---

## Extending

- Add deployment scripts (Flask/FastAPI).
- Integrate automated retraining (CI/CD).
- Add data validation and monitoring.

---

## Notes

- If you change the data filename or location, update the `--data_path` argument accordingly in your scripts.
- Make sure the feature order for prediction matches the order used during training.

