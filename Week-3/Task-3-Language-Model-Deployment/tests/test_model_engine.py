"""
NEXUS NLP — Automated Unit Test Suite for Inference Engine & Evaluation Metrics
Author: Subham Gupta (ShadowFox ML Engineer Intern)
Week 3: Task 3 - Language Model Deployment & Intelligence Evaluation Lab
"""

import os
import pytest
from src.model_engine import NexusModelEngine, GenerationResult
from src.metrics import (
    normalize_text,
    compute_exact_match,
    compute_token_f1,
    evaluate_qa,
    compute_rouge,
    compute_sacrebleu,
    compute_classification_metrics
)

# Shared fixture for loaded engine
@pytest.fixture(scope="module")
def engine():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(base_dir, "model_cache")
    eng = NexusModelEngine(cache_dir=cache_dir, device="cpu")
    eng.load()
    return eng


# ==============================================================================
# Model Loading & Architectural Specs Tests
# ==============================================================================

def test_engine_initialization(engine):
    assert engine.is_loaded is True
    assert engine.model is not None
    assert engine.tokenizer is not None
    specs = engine.get_specs()
    assert specs["device"] == "cpu"
    assert specs["config"]["num_layers"] == 8
    assert specs["config"]["num_decoder_layers"] == 8
    assert specs["config"]["d_model"] == 512
    assert specs["config"]["feed_forward_proj"] == "gated-gelu"
    assert specs["config"]["total_parameters"] > 70_000_000


# ==============================================================================
# Tokenization & Generation Tests
# ==============================================================================

def test_tokenization_and_detokenization(engine):
    prompt = "Translate English to German: Artificial Intelligence is transforming society."
    tokens = engine.tokenizer(prompt, return_tensors="pt")
    assert "input_ids" in tokens
    assert "attention_mask" in tokens
    decoded = engine.tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
    assert "Artificial Intelligence" in decoded


def test_greedy_generation(engine):
    prompt = "Classify sentiment as positive or negative: The service was quick and very courteous."
    result = engine.generate(prompt, strategy="greedy", max_new_tokens=16)
    assert isinstance(result, GenerationResult)
    assert len(result.generated_text) > 0
    assert result.decoding_strategy == "greedy"
    assert result.latency_ms > 0
    assert result.output_tokens > 0
    assert result.tokens_per_second > 0
    assert "positive" in result.generated_text.lower()


def test_beam_search_generation(engine):
    prompt = "Summarize: The Earth revolves around the Sun once every 365.25 days, creating our four seasons."
    result = engine.generate(prompt, strategy="beam_search", num_beams=3, max_new_tokens=32)
    assert isinstance(result, GenerationResult)
    assert len(result.generated_text) > 0
    assert result.decoding_strategy == "beam_search"
    assert result.parameters["num_beams"] == 3


def test_sampling_generation(engine):
    prompt = "Write a creative analogy for neural networks:"
    result = engine.generate(
        prompt,
        strategy="sampling",
        temperature=0.8,
        top_k=40,
        top_p=0.92,
        max_new_tokens=32
    )
    assert isinstance(result, GenerationResult)
    assert len(result.generated_text) > 0
    assert result.decoding_strategy == "sampling"
    assert result.parameters["do_sample"] is True
    assert result.parameters["temperature"] == 0.8


# ==============================================================================
# Input Validation & Defensive Handling Tests
# ==============================================================================

def test_empty_prompt_validation(engine):
    with pytest.raises(ValueError, match="Prompt must be a non-empty string"):
        engine.generate("", strategy="greedy")

    with pytest.raises(ValueError, match="Prompt must be a non-empty string"):
        engine.generate("    ", strategy="greedy")


def test_invalid_max_new_tokens(engine):
    with pytest.raises(ValueError, match="max_new_tokens must be an integer"):
        engine.generate("Hello", max_new_tokens=-5)

    with pytest.raises(ValueError, match="max_new_tokens must be an integer"):
        engine.generate("Hello", max_new_tokens=5000)


def test_invalid_strategy(engine):
    with pytest.raises(ValueError, match="Unsupported strategy"):
        engine.generate("Hello", strategy="quantum_sampling")


def test_sampling_param_with_deterministic_strategy(engine):
    # Enforce that temperature is not combined with do_sample=False
    with pytest.raises(ValueError, match="Temperature.*specified with do_sample=False"):
        engine.validate_inputs(
            prompt="Test",
            max_new_tokens=32,
            strategy="greedy",
            do_sample=False,
            temperature=0.7,
            top_k=None,
            top_p=None,
            num_beams=1,
            repetition_penalty=1.0
        )


def test_invalid_temperature_bounds(engine):
    with pytest.raises(ValueError, match="temperature must be between 0.01 and 3.0"):
        engine.validate_inputs(
            prompt="Test",
            max_new_tokens=32,
            strategy="sampling",
            do_sample=True,
            temperature=0.0,
            top_k=50,
            top_p=0.9,
            num_beams=1,
            repetition_penalty=1.0
        )


# ==============================================================================
# Metrics Calculation Tests
# ==============================================================================

def test_normalization():
    raw = " The, QUICK! brown Fox... "
    norm = normalize_text(raw)
    assert norm == "quick brown fox"


def test_qa_exact_match_and_f1():
    assert compute_exact_match("Albert Einstein", "albert einstein") == 1.0
    assert compute_exact_match("The Einstein", "Einstein") == 1.0  # article removed
    assert compute_exact_match("Isaac Newton", "Albert Einstein") == 0.0

    # Token F1
    pred = "the quick brown fox"
    ref = "quick brown dog"
    # common: {"quick", "brown"} (2), pred len: 3, ref len: 3 -> p=2/3, r=2/3, f1=2/3
    f1 = compute_token_f1(pred, ref)
    assert round(f1, 2) == 0.67


def test_rouge_calculation():
    preds = ["The cat sat on the mat."]
    refs = ["A cat was sitting on a mat."]
    scores = compute_rouge(preds, refs)
    assert "rouge1" in scores
    assert "rouge2" in scores
    assert "rougeL" in scores
    assert scores["rouge1"] > 0.0
    assert scores["rougeL"] > 0.0


def test_sacrebleu_calculation():
    preds = ["Good morning, welcome to our presentation."]
    refs = ["Good morning, welcome to our presentation."]
    res = compute_sacrebleu(preds, refs)
    assert res["bleu"] == 100.0


def test_classification_accuracy():
    preds = ["positive", "Negative", "Neutral"]
    refs = ["Positive", "negative", "neutral"]
    metrics = compute_classification_metrics(preds, refs)
    assert metrics["accuracy"] == 100.0
    assert metrics["correct"] == 3
