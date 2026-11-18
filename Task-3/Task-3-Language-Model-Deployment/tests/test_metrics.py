"""
NEXUS NLP — Comprehensive Unit Test Suite for NLP Evaluation Metrics
Tests exact matching, token F1, ROUGE, SacreBLEU, and hardened classification logic.
Author: Subham Gupta (ShadowFox ML Engineer Intern)
Task 3 - Language Model Deployment & Intelligence Evaluation Lab
"""

import pytest
from src.metrics import (
    normalize_text,
    normalize_label,
    extract_predicted_label,
    compute_exact_match,
    compute_token_f1,
    evaluate_qa,
    compute_rouge,
    compute_sacrebleu,
    compute_classification_metrics
)


# ==============================================================================
# Text Normalization & Label Normalization Tests
# ==============================================================================

def test_normalize_text_standard():
    raw = "The quick brown fox jumps over the lazy dog."
    norm = normalize_text(raw)
    assert norm == "quick brown fox jumps over lazy dog"


def test_normalize_label_variants():
    assert normalize_label("  Positive  ") == "positive"
    assert normalize_label('"Positive."') == "positive"
    assert normalize_label("'Negative'") == "negative"
    assert normalize_label("Technology\n") == "technology"
    assert normalize_label("") == ""
    assert normalize_label(None) == ""


# ==============================================================================
# Strict Classification Label Extraction & Substring Trap Tests
# ==============================================================================

def test_extract_exact_label():
    allowed = ["positive", "negative", "neutral"]
    assert extract_predicted_label("positive", allowed) == "positive"
    assert extract_predicted_label("Negative", allowed) == "negative"
    assert extract_predicted_label("  NEUTRAL.  ", allowed) == "neutral"


def test_extract_label_with_extra_text():
    allowed = ["positive", "negative"]
    assert extract_predicted_label("The sentiment is positive.", allowed) == "positive"
    assert extract_predicted_label("Overall rating: Negative!", allowed) == "negative"


def test_extract_label_negation_rejection():
    allowed = ["positive", "negative"]
    # "not positive" should NEVER match "positive"
    assert extract_predicted_label("This movie is not positive.", allowed) is None
    assert extract_predicted_label("The experience was no positive feeling.", allowed) is None
    assert extract_predicted_label("Never positive about this outcome.", allowed) is None


def test_substring_trap_formal_informal():
    """
    Critical vulnerability test:
    'formal' is a literal substring of 'informal'.
    'informal' MUST NEVER match 'formal', and 'formal' MUST NEVER match 'informal'.
    """
    allowed = ["formal", "informal"]
    assert extract_predicted_label("informal", allowed) == "informal"
    assert extract_predicted_label("formal", allowed) == "formal"
    assert extract_predicted_label("The tone is informal.", allowed) == "informal"
    assert extract_predicted_label("The tone is formal.", allowed) == "formal"

    # Strict metric evaluation
    res = compute_classification_metrics(["informal"], ["formal"], allowed_labels=allowed)
    assert res["accuracy"] == 0.0
    assert res["correct"] == 0

    res_reverse = compute_classification_metrics(["formal"], ["informal"], allowed_labels=allowed)
    assert res_reverse["accuracy"] == 0.0
    assert res_reverse["correct"] == 0


def test_ambiguous_prediction_rejection():
    allowed = ["positive", "negative"]
    # If both labels are present, it must be flagged as ambiguous
    assert extract_predicted_label("It has both positive and negative elements.", allowed) is None


def test_unmatched_garbage_prediction():
    allowed = ["technology", "sports", "finance"]
    assert extract_predicted_label("Random unrelated output 12345", allowed) is None
    assert extract_predicted_label("", allowed) is None


# ==============================================================================
# Multi-Class Evaluation & Metric Aggregation Tests
# ==============================================================================

def test_classification_metrics_perfect_batch():
    preds = ["Positive", "negative", "Technology", "Sports"]
    refs = ["positive", "Negative", "technology", "sports"]
    allowed = ["positive", "negative", "technology", "sports"]

    metrics = compute_classification_metrics(preds, refs, allowed_labels=allowed)
    assert metrics["accuracy"] == 100.0
    assert metrics["macro_f1"] == 100.0
    assert metrics["correct"] == 4
    assert metrics["total"] == 4
    assert metrics["unmatched"] == 0


def test_classification_metrics_mixed_batch():
    preds = ["positive", "negative", "Business", "Sports", "Technology", "Informal"]
    refs = ["positive", "negative", "Finance", "Sports", "Technology", "Formal"]
    allowed = [
        ["positive", "negative"],
        ["positive", "negative"],
        ["Technology", "Sports", "Finance"],
        ["Technology", "Sports", "Finance"],
        ["Technology", "Sports", "Finance"],
        ["Formal", "Informal"]
    ]

    metrics = compute_classification_metrics(preds, refs, allowed_labels=allowed)
    # 4 correct: positive, negative, Sports, Technology
    # 2 incorrect: Business (unmatched in Finance/Sports/Tech), Informal (does not match Formal)
    assert metrics["correct"] == 4
    assert metrics["total"] == 6
    assert round(metrics["accuracy"], 2) == 66.67
    assert metrics["unmatched"] == 1  # 'Business' is not in ['Technology', 'Sports', 'Finance']


def test_classification_length_mismatch():
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_classification_metrics(["positive"], ["positive", "negative"])


def test_classification_empty_batch():
    metrics = compute_classification_metrics([], [])
    assert metrics["accuracy"] == 0.0
    assert metrics["total"] == 0
    assert metrics["correct"] == 0


# ==============================================================================
# QA, ROUGE & SacreBLEU Suite Tests
# ==============================================================================

def test_exact_match_and_f1():
    assert compute_exact_match("Neil Armstrong", "neil armstrong") == 1.0
    assert compute_exact_match("The Neil Armstrong", "Neil Armstrong") == 1.0
    assert compute_exact_match("Buzz Aldrin", "Neil Armstrong") == 0.0

    qa_res = evaluate_qa(
        ["Neil Armstrong", "2017"],
        ["Neil Armstrong", "2017"]
    )
    assert qa_res["exact_match"] == 100.0
    assert qa_res["f1"] == 100.0


def test_rouge_evaluation():
    preds = ["The James Webb Space Telescope observes infrared light from distant stars."]
    refs = ["James Webb Space Telescope observes infrared light from early stars and galaxies."]
    res = compute_rouge(preds, refs)
    assert res["rouge1"] > 50.0
    assert res["rouge2"] > 20.0
    assert res["rougeL"] > 50.0
    assert res["count"] == 1


def test_sacrebleu_evaluation():
    preds = ["Good morning, welcome to our presentation."]
    refs = ["Good morning, welcome to our presentation."]
    res = compute_sacrebleu(preds, refs)
    assert res["bleu"] == 100.0
    assert res["count"] == 1
