"""Command-line interface: generate data, train a model, and score new customers."""

from __future__ import annotations

import argparse
import sys

import joblib
import pandas as pd

from churn.data import CATEGORICAL, NUMERIC, make_synthetic
from churn.train import save, train


def _cmd_generate(args) -> int:
    df = make_synthetic(args.rows, args.seed)
    df.to_csv(args.out, index=False)
    print(f'wrote {len(df)} rows to {args.out}')
    return 0


def _cmd_train(args) -> int:
    df = pd.read_csv(args.data)
    pipeline, metrics, report = train(df, model_name=args.model, seed=args.seed)
    out = save(pipeline, metrics, args.out_dir)
    print(report)
    print(f'CV ROC AUC: {metrics.cv_roc_auc_mean:.3f} +/- {metrics.cv_roc_auc_std:.3f}')
    print(f'Test ROC AUC: {metrics.test_roc_auc:.3f}')
    print(f'artifacts saved to {out}')
    return 0


def _cmd_predict(args) -> int:
    pipeline = joblib.load(args.model)
    df = pd.read_csv(args.data)
    X = df[CATEGORICAL + NUMERIC]
    scored = df.copy()
    scored['churn_probability'] = pipeline.predict_proba(X)[:, 1].round(4)
    scored.to_csv(args.out, index=False)
    print(f'scored {len(scored)} customers -> {args.out}')
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='churn', description='Customer churn pipeline.')
    sub = parser.add_subparsers(dest='command', required=True)

    g = sub.add_parser('generate', help='write a synthetic customer CSV')
    g.add_argument('--rows', type=int, default=5000)
    g.add_argument('--seed', type=int, default=42)
    g.add_argument('--out', default='customers.csv')
    g.set_defaults(func=_cmd_generate)

    t = sub.add_parser('train', help='train and evaluate a model')
    t.add_argument('--data', default='customers.csv')
    t.add_argument('--model', choices=['logistic', 'random_forest'], default='random_forest')
    t.add_argument('--seed', type=int, default=42)
    t.add_argument('--out-dir', default='artifacts')
    t.set_defaults(func=_cmd_train)

    p = sub.add_parser('predict', help='score customers with a trained model')
    p.add_argument('--model', default='artifacts/model.joblib')
    p.add_argument('--data', required=True)
    p.add_argument('--out', default='scored.csv')
    p.set_defaults(func=_cmd_predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
