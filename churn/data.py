"""Synthetic customer-churn data generator.

The generator uses a known logistic model so that the pipeline can be tested
against a ground truth. Real data can be used by passing any CSV with the same
columns and a binary 'churned' column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL = ['contract', 'internet_service', 'payment_method']
NUMERIC = ['tenure_months', 'monthly_charges', 'total_charges', 'support_calls', 'senior_citizen']
TARGET = 'churned'

CONTRACTS = ['month-to-month', 'one-year', 'two-year']
INTERNET = ['dsl', 'fiber', 'none']
PAYMENTS = ['electronic-check', 'mailed-check', 'bank-transfer', 'credit-card']


def make_synthetic(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a customer table where churn depends on tenure, contract, charges and support calls."""
    if n_rows <= 0:
        raise ValueError('n_rows must be positive')

    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 73, size=n_rows)
    contract = rng.choice(CONTRACTS, size=n_rows, p=[0.55, 0.25, 0.20])
    internet = rng.choice(INTERNET, size=n_rows, p=[0.35, 0.45, 0.20])
    payment = rng.choice(PAYMENTS, size=n_rows)
    senior = rng.binomial(1, 0.16, size=n_rows)

    base = np.where(internet == 'fiber', 85.0, np.where(internet == 'dsl', 55.0, 25.0))
    monthly = np.clip(base + rng.normal(0, 10, size=n_rows), 18, None).round(2)
    total = (monthly * tenure * rng.uniform(0.9, 1.05, size=n_rows)).round(2)
    support = rng.poisson(1.2, size=n_rows)

    # Ground-truth churn logit: short tenure, month-to-month, high charges and
    # frequent support calls all increase the chance of leaving.
    logit = (
        1.2
        - 0.06 * tenure
        + 1.4 * (contract == 'month-to-month')
        - 0.8 * (contract == 'two-year')
        + 0.015 * (monthly - 65)
        + 0.35 * support
        + 0.3 * senior
        + 0.4 * (payment == 'electronic-check')
    )
    prob = 1 / (1 + np.exp(-logit))
    churned = rng.binomial(1, prob)

    return pd.DataFrame({
        'tenure_months': tenure,
        'monthly_charges': monthly,
        'total_charges': total,
        'support_calls': support,
        'senior_citizen': senior,
        'contract': contract,
        'internet_service': internet,
        'payment_method': payment,
        TARGET: churned,
    })


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) after checking the required columns are present."""
    missing = [c for c in CATEGORICAL + NUMERIC + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f'missing required columns: {missing}')
    return df[CATEGORICAL + NUMERIC], df[TARGET].astype(int)
