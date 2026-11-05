"""Step 14 Streamlit app contract checks. No test.csv is opened or used."""
from pathlib import Path
import importlib.util
import math

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "app/streamlit_app.py"
spec = importlib.util.spec_from_file_location("p4_streamlit_app", MODULE)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


@pytest.fixture
def sample():
    frame = pd.read_csv(ROOT / "app/sample_leads.csv")
    return app.validate_demo_csv(frame)


class PredictableModel:
    """Safe fake scoring model. Does not open a trained model or test data."""
    def predict_proba(self, features):
        n = len(features)
        scores = np.linspace(0.95, 0.05, n)
        return np.column_stack([1 - scores, scores])


def test_sample_data_is_valid_and_has_20_rows(sample):
    assert len(sample) == 20
    assert list(sample.columns) == app.FEATURES


@pytest.mark.parametrize("capacity,expected", [(10, 2), (20, 4), (100, 20)])
def test_capacity_counts_and_export_contract(sample, capacity, expected):
    ranked, count = app.rank_leads(PredictableModel(), sample, capacity)
    assert count == expected
    assert int(ranked["selected_for_review"].sum()) == expected
    assert sorted(ranked["priority_rank"]) == list(range(1, 21))
    assert ranked.iloc[0]["priority_rank"] == 1
    assert "predicted_probability" not in ranked.columns
    assert "target" not in ranked.columns
    assert "rank_percentile" in ranked.columns


def test_missing_feature_rejected(sample):
    with pytest.raises(ValueError, match="exactly these seven"):
        app.validate_demo_csv(sample.drop(columns=["loan"]))


def test_extra_target_column_rejected(sample):
    frame = sample.copy()
    frame["y"] = "yes"
    with pytest.raises(ValueError, match="exactly these seven"):
        app.validate_demo_csv(frame)


def test_unknown_category_rejected(sample):
    frame = sample.copy()
    frame.loc[0, "poutcome"] = "invented"
    with pytest.raises(ValueError, match="poutcome"):
        app.validate_demo_csv(frame)


def test_invalid_prior_contact_flag_rejected(sample):
    frame = sample.copy()
    frame.loc[0, "prior_contacted"] = 2
    with pytest.raises(ValueError, match="prior_contacted"):
        app.validate_demo_csv(frame)


def test_invalid_prior_contact_days_rejected(sample):
    frame = sample.copy()
    frame.loc[0, "prior_contacted"] = 0
    frame.loc[0, "days_since_previous"] = 12
    with pytest.raises(ValueError, match="leave days_since_previous blank"):
        app.validate_demo_csv(frame)


def test_oversized_row_count_rejected(sample):
    oversized = pd.concat([sample] * 501, ignore_index=True)
    assert len(oversized) > 10000
    with pytest.raises(ValueError, match="10,000"):
        app.validate_demo_csv(oversized)


def test_locked_model_hash_matches_saved_final_evaluation():
    lock, metrics, digest = app.locked_artifacts()
    assert digest == lock["sha256"] == metrics["model_sha256"]
    assert lock["features_in_order"] == app.FEATURES


def test_real_locked_model_scores_synthetic_sample(sample):
    _, _, digest = app.locked_artifacts()
    model = app.load_model(digest)
    ranked, count = app.rank_leads(model, sample, 10)
    assert len(ranked) == 20
    assert count == 2
    assert ranked["priority_rank"].nunique() == 20
