# churn-prediction-pipeline

A churn prediction project in Python with scikit-learn. It creates synthetic customer data, trains a model, checks how well it does on held-out data, and scores new customers from the command line.

The data is synthetic, so you can run everything without downloading anything. The generator uses a known logistic model, which makes it a useful sanity check: a good pipeline should find the pattern that was built in. You can also use your own CSV, as long as it has the same columns and a binary `churned` column.

## How it works

1. **Data:** `churn/data.py` creates customers with tenure, charges, support calls, contract type, internet service and payment method. Short tenure, month-to-month contracts, high charges and lots of support calls make churn more likely.
2. **Preprocessing and model:** `churn/model.py` puts scaling, one-hot encoding and the classifier into a single scikit-learn `Pipeline`. The preprocessing is fitted only on training data, so cross-validation doesn't leak information from the test folds.
3. **Evaluation:** `churn/train.py` makes a stratified train/test split, reports 5-fold cross-validated ROC AUC on the training set, then reports ROC AUC and a classification report on the test set.
4. **Saving:** the fitted pipeline is saved with joblib, and the metrics go into a JSON file.
5. **Scoring:** `churn predict` adds a `churn_probability` column to any CSV that has the feature columns.

## Results on synthetic data (5,000 rows, seed 42)

| Model | CV ROC AUC (mean ± std) | Test ROC AUC |
| --- | --- | --- |
| Logistic regression | 0.829 ± 0.010 | 0.839 |
| Random forest | 0.810 ± 0.010 | 0.827 |

Logistic regression comes out ahead, which is probably because the synthetic labels come from a logistic model. I wouldn't read much into the gap. On real data, the ranking could be different, so try both.

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

`predict` works on any CSV with the feature columns. It doesn't evaluate the model. For that, use the held-out metrics printed by `train`.

## Tests

```bash
pytest -q
```

The tests cover data generation and reproducibility, input validation, model selection, a minimum quality check on synthetic data, and a full generate, train and predict run through the CLI.

## Limitations

- The data is synthetic, so accuracy on real customers will be different. Feature engineering would matter more on real data.
- There's no hyperparameter search and no model registry.

## Layout

| Path | What it does |
| --- | --- |
| `churn/data.py` | Synthetic data generator and column checks |
| `churn/model.py` | Preprocessing and model pipeline |
| `churn/train.py` | Training, evaluation and saving |
| `churn/cli.py` | Command-line interface |
| `tests/test_pipeline.py` | Tests |

## License

MIT
