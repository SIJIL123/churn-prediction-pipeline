# churn-prediction-pipeline

An end-to-end customer churn pipeline in Python: synthetic data generation, a preprocessing and model pipeline built with scikit-learn, cross-validated evaluation, saved artifacts, and a command-line interface for training and scoring.

The data is synthetic so the project runs without a download. The generator uses a known logistic model, so you can check that the pipeline recovers real structure. The same code works on any CSV with the same columns and a binary `churned` column.

## Pipeline

1. **Data:** `churn/data.py` generates customers with tenure, charges, support calls, contract type, internet service, and payment method. Churn is drawn from a logistic model in which short tenure, month-to-month contracts, high charges, and frequent support calls raise the probability of leaving.
2. **Preprocessing and model:** `churn/model.py` wraps a `ColumnTransformer` (standard scaling for numeric columns, one-hot encoding for categorical columns) and a classifier in one `Pipeline`. Preprocessing is fit only on training folds, so there is no leakage into cross-validation.
3. **Evaluation:** `churn/train.py` makes a stratified train/test split, reports 5-fold cross-validated ROC AUC on the training set, then reports ROC AUC and a classification report on the held-out test set.
4. **Persistence:** the fitted pipeline is saved with joblib and the metrics are written to JSON.
5. **Inference:** `churn predict` adds a `churn_probability` column to any CSV with the feature columns.

## Results (synthetic data, 5,000 rows, seed 42)

| Model | CV ROC AUC (mean ± std) | Test ROC AUC |
| --- | --- | --- |
| Logistic regression | 0.829 ± 0.010 | 0.839 |
| Random forest | 0.810 ± 0.010 | 0.827 |

Logistic regression scores higher because the synthetic labels come from a logistic model. On real data the ranking may change, so run both and compare.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Generate synthetic data
python -m churn.cli generate --rows 5000 --out customers.csv

# 2. Train and evaluate (logistic or random_forest)
python -m churn.cli train --data customers.csv --model logistic --out-dir artifacts

# 3. Score customers
python -m churn.cli predict --model artifacts/model.joblib --data customers.csv --out scored.csv
```

## Tests

```bash
pytest -q
```

The tests cover data generation and reproducibility, input validation, model selection, minimum model quality on synthetic data, and a full generate, train, predict round trip through the CLI.

## Limitations

- The data is synthetic. Performance on real churn data will differ, and feature engineering would matter more.
- `predict` scores the same file it reads, so it doesn't check model performance on unseen data. Use the held-out metrics from `train` for that.
- There's no hyperparameter search or model registry.

## Project layout

| Path | Purpose |
| --- | --- |
| `churn/data.py` | Synthetic data generator and column checks |
| `churn/model.py` | Preprocessing and model pipeline |
| `churn/train.py` | Training, evaluation, and saving |
| `churn/cli.py` | Command-line interface |
| `tests/test_pipeline.py` | pytest suite |

## License

MIT
