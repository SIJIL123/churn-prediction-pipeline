"""Model construction: preprocessing plus classifier in a single sklearn Pipeline."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.data import CATEGORICAL, NUMERIC

MODELS = {
    'logistic': lambda seed: LogisticRegression(max_iter=1000, class_weight='balanced'),
    'random_forest': lambda seed: RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, class_weight='balanced', random_state=seed, n_jobs=-1
    ),
}


def build_pipeline(model_name: str = 'random_forest', seed: int = 42) -> Pipeline:
    """Build a preprocessing + model pipeline. Preprocessing is fit only on training folds."""
    if model_name not in MODELS:
        raise ValueError(f'unknown model {model_name!r}; choose from {sorted(MODELS)}')

    preprocess = ColumnTransformer([
        ('num', StandardScaler(), NUMERIC),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
    ])
    return Pipeline([
        ('preprocess', preprocess),
        ('model', MODELS[model_name](seed)),
    ])
