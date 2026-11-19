"""
Unit tests for leakage-free model selection in Task 2.
Verifies that model selection strictly relies on cross-validation metrics and forbids test-set metrics.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.train import select_best_model_cv

def test_model_selection_rejects_test_metrics():
    """Verify that attempting to select a model using test-set metrics raises ValueError."""
    dummy_results = [
        {"Model": "Model_A", "CV_R2_Mean": 0.85, "Test_R2": 0.90},
        {"Model": "Model_B", "CV_R2_Mean": 0.88, "Test_R2": 0.80}
    ]

    with pytest.raises(ValueError, match="Test-set leakage prevented"):
        select_best_model_cv(dummy_results, metric="Test_R2")

    with pytest.raises(ValueError, match="Test-set leakage prevented"):
        select_best_model_cv(dummy_results, metric="test_mae")

    with pytest.raises(ValueError, match="Test-set leakage prevented"):
        select_best_model_cv(dummy_results, metric="Test_RMSE")

    with pytest.raises(ValueError, match="Test-set leakage prevented"):
        select_best_model_cv(dummy_results, metric="TEST_SCORE")

    with pytest.raises(ValueError, match="Test-set leakage prevented"):
        select_best_model_cv(dummy_results, metric="held_out_test_eval")

def test_model_selection_strictly_uses_cv_metrics():
    """
    Verify that candidate selection depends 100% on CV metrics and is invariant to test scores.
    Even if Model B has a much higher Test R2, Model A must be selected because its CV R2 is higher.
    """
    candidates = [
        {"Model": "Model_A", "CV_R2_Mean": 0.9056, "Test_R2": 0.7143},
        {"Model": "Model_B", "CV_R2_Mean": 0.8973, "Test_R2": 0.8119}
    ]

    selected = select_best_model_cv(candidates, metric="CV_R2_Mean", mode="max")
    assert selected == "Model_A", f"Expected Model_A (CV R2 0.9056), got {selected}"

def test_model_selection_invariant_to_arbitrary_test_scores():
    """Verify mutating test metrics to arbitrary extremes never changes the selection outcome."""
    candidates = [
        {"Model": "Model_X", "CV_R2_Mean": 0.9100, "Test_R2": -50.0},
        {"Model": "Model_Y", "CV_R2_Mean": 0.8900, "Test_R2": 0.9999}
    ]
    # Model_X has lower test R2 but higher CV R2 -> Model_X MUST be chosen
    selected = select_best_model_cv(candidates, metric="CV_R2_Mean", mode="max")
    assert selected == "Model_X"

def test_model_selection_minimization_mode():
    """Verify that mode='min' correctly selects the candidate with the lowest CV error."""
    candidates = [
        {"Model": "Gradient Boosting", "CV_MAE_Mean": 0.6871},
        {"Model": "Linear Regression", "CV_MAE_Mean": 1.1032},
        {"Model": "Ridge Regression", "CV_MAE_Mean": 1.1004}
    ]

    selected = select_best_model_cv(candidates, metric="CV_MAE_Mean", mode="min")
    assert selected == "Gradient Boosting", f"Expected Gradient Boosting, got {selected}"

def test_model_selection_validation_errors():
    """Verify that invalid inputs (empty lists, missing metric keys) raise appropriate errors."""
    with pytest.raises(ValueError, match="Candidate results list cannot be empty"):
        select_best_model_cv([], metric="CV_R2_Mean")

    incomplete_candidates = [
        {"Model": "Model_A", "CV_R2_Mean": 0.85},
        {"Model": "Model_B"}  # Missing metric
    ]
    with pytest.raises(KeyError, match="Metric 'CV_R2_Mean' not found"):
        select_best_model_cv(incomplete_candidates, metric="CV_R2_Mean")

    with pytest.raises(ValueError, match="Invalid mode"):
        select_best_model_cv(incomplete_candidates[:1], metric="CV_R2_Mean", mode="median")

def test_single_test_evaluation_protocol_integrity():
    """
    Verify that in the regenerated benchmark CSV:
    1. Exactly one model (the winning selected model) has a non-null Test_R2.
    2. All unselected candidate models have NaN/null in Test_R2.
    Proves that candidate models were not evaluated repeatedly on the held-out test set.
    """
    csv_path = ROOT_DIR / "reports" / "experiment_results.csv"
    if not csv_path.exists():
        pytest.skip("experiment_results.csv has not been generated yet.")

    df = pd.read_csv(csv_path)
    assert "Test_R2" in df.columns
    assert "Model" in df.columns

    non_null_test_evals = df[df["Test_R2"].notna()]
    assert len(non_null_test_evals) == 1, (
        f"Expected exactly 1 model evaluated on the held-out test set, but found {len(non_null_test_evals)}. "
        f"Candidate baseline models must not be evaluated on the test set to avoid multiple-testing leakage."
    )
    winning_model = non_null_test_evals.iloc[0]["Model"]
    assert "Optimized" in winning_model or "Gradient Boosting" in winning_model
