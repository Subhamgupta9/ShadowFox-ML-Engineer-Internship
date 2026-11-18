"""
NEXUS NLP — Automated Unit Tests for Streamlit App Components & Data Loading
Author: Subham Gupta (ShadowFox ML Engineer Intern)
Week 3: Task 3 - Language Model Deployment & Intelligence Evaluation Lab
"""

import os
import pytest
import pandas as pd

from app.components.tab_benchmark import load_benchmark_data
from app.components.tab_lab import TASK_PRESETS
from app.components.tab_comparator import COMPARATOR_PRESETS
from src.model_engine import NexusModelEngine


@pytest.fixture(scope="module")
def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ==============================================================================
# Benchmark Data Ingestion & Integrity Tests
# ==============================================================================

def test_load_benchmark_data(project_root):
    reports_dir = os.path.join(project_root, "reports")
    metrics_data, df_records = load_benchmark_data(reports_dir)

    assert isinstance(metrics_data, dict), "metrics_data should be a dictionary"
    assert "summarization" in metrics_data
    assert "question_answering" in metrics_data
    assert "classification" in metrics_data
    assert "translation" in metrics_data
    assert "reasoning" in metrics_data

    assert isinstance(df_records, pd.DataFrame), "df_records should be a DataFrame"
    assert not df_records.empty, "df_records should not be empty"
    assert len(df_records) == 45, f"Expected 45 benchmark records, got {len(df_records)}"

    # Check essential column existence
    expected_cols = {"task", "prompt", "generated", "reference", "strategy", "latency_ms", "tokens_per_second"}
    assert expected_cols.issubset(set(df_records.columns))


# ==============================================================================
# Interactive Lab Presets Configuration Tests
# ==============================================================================

def test_task_presets_coverage():
    expected_tasks = [
        "Summarization",
        "Question Answering (Context-Grounded)",
        "Zero-Shot Classification",
        "Machine Translation",
        "Instruction Following",
        "Reasoning-Style Prompts"
    ]
    for task in expected_tasks:
        assert task in TASK_PRESETS, f"Missing preset for task: {task}"
        preset = TASK_PRESETS[task]
        assert "prompt" in preset and len(preset["prompt"].strip()) > 0
        assert "recommended_strategy" in preset
        assert preset["recommended_strategy"] in NexusModelEngine.SUPPORTED_STRATEGIES
        assert preset["max_new_tokens"] > 0


# ==============================================================================
# Decoding Comparator Presets Tests
# ==============================================================================

def test_comparator_presets():
    assert isinstance(COMPARATOR_PRESETS, list)
    assert len(COMPARATOR_PRESETS) >= 3
    for p in COMPARATOR_PRESETS:
        assert isinstance(p, str) and len(p.strip()) > 0


# ==============================================================================
# Engine Input Validation Edge Case Tests
# ==============================================================================

def test_repetition_penalty_bounds(project_root):
    cache_dir = os.path.join(project_root, "model_cache")
    engine = NexusModelEngine(cache_dir=cache_dir, device="cpu")

    # Repetition penalty < 1.0 should raise ValueError
    with pytest.raises(ValueError, match="repetition_penalty must be between 1.0 and 5.0"):
        engine.validate_inputs(
            prompt="Hello",
            max_new_tokens=32,
            strategy="greedy",
            do_sample=False,
            temperature=None,
            top_k=None,
            top_p=None,
            num_beams=1,
            repetition_penalty=0.5
        )

    # Repetition penalty > 5.0 should raise ValueError
    with pytest.raises(ValueError, match="repetition_penalty must be between 1.0 and 5.0"):
        engine.validate_inputs(
            prompt="Hello",
            max_new_tokens=32,
            strategy="greedy",
            do_sample=False,
            temperature=None,
            top_k=None,
            top_p=None,
            num_beams=1,
            repetition_penalty=6.0
        )
