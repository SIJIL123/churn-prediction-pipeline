import pandas as pd
import pytest

from churn.cli import main
from churn.data import CATEGORICAL, NUMERIC, TARGET, make_synthetic, split_features_target
from churn.model import build_pipeline
from churn.train import train


def test_synthetic_data_shape_and_columns():
    df = make_synthetic(500, seed=1)
    assert len(df) == 500
    assert set(CATEGORICAL + NUMERIC + [TARGET]).issubset(df.columns)
    assert set(df[TARGET].unique()) <= {0, 1}


def test_synthetic_data_is_reproducible():
    a = make_synthetic(200, seed=7)
    b = make_synthetic(200, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_invalid_row_count_rejected():
    with pytest.raises(ValueError):
        make_synthetic(0)


def test_split_rejects_missing_columns():
    with pytest.raises(ValueError, match='missing required columns'):
        split_features_target(pd.DataFrame({'tenure_months': [1]}))


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        build_pipeline('not-a-model')


def test_logistic_beats_chance_on_synthetic_data():
    df = make_synthetic(3000, seed=0)
    _, metrics, _ = train(df, model_name='logistic', seed=0, cv_folds=3)
    assert metrics.test_roc_auc > 0.7


def test_random_forest_beats_chance_on_synthetic_data():
    df = make_synthetic(3000, seed=0)
    _, metrics, report = train(df, model_name='random_forest', seed=0, cv_folds=3)
    assert metrics.test_roc_auc > 0.7
    assert 'precision' in report


def test_cli_generate_train_predict_round_trip(tmp_path):
    data = tmp_path / 'customers.csv'
    artifacts = tmp_path / 'artifacts'
    scored = tmp_path / 'scored.csv'

    assert main(['generate', '--rows', '800', '--out', str(data)]) == 0
    assert main(['train', '--data', str(data), '--model', 'logistic', '--out-dir', str(artifacts)]) == 0
    assert (artifacts / 'model.joblib').exists()
    assert (artifacts / 'metrics.json').exists()

    assert main(['predict', '--model', str(artifacts / 'model.joblib'),
                 '--data', str(data), '--out', str(scored)]) == 0
    out = pd.read_csv(scored)
    assert 'churn_probability' in out.columns
    assert out['churn_probability'].between(0, 1).all()
