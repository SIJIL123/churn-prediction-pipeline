"""Training and evaluation: stratified split, cross-validation, held-out metrics, model persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from churn.data import split_features_target
from churn.model import build_pipeline


@dataclass
class Metrics:
    model: str
    cv_roc_auc_mean: float
    cv_roc_auc_std: float
    test_roc_auc: float
    test_positive_rate: float


def train(df: pd.DataFrame, model_name: str = 'random_forest', seed: int = 42,
          test_size: float = 0.2, cv_folds: int = 5):
    """Fit the model and return (fitted_pipeline, metrics, classification_report_text)."""
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    pipeline = build_pipeline(model_name, seed)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')

    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, pipeline.predict(X_test), digits=3)

    metrics = Metrics(
        model=model_name,
        cv_roc_auc_mean=float(cv_scores.mean()),
        cv_roc_auc_std=float(cv_scores.std()),
        test_roc_auc=float(roc_auc_score(y_test, proba)),
        test_positive_rate=float(y_test.mean()),
    )
    return pipeline, metrics, report


def save(pipeline, metrics: Metrics, out_dir: str | Path) -> Path:
    """Persist the pipeline and a JSON metrics file to out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out / 'model.joblib')
    (out / 'metrics.json').write_text(json.dumps(asdict(metrics), indent=2))
    return out
