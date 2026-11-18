"""
NEXUS NLP — Automated Unit Test Suite for Environment & Dependencies
"""

import sys
import pytest

def test_python_version():
    assert sys.version_info >= (3, 11), "Python version must be 3.11+"
    assert sys.prefix != sys.base_prefix, "Must run inside dedicated virtual environment"

def test_torch_cpu():
    import torch
    assert not torch.cuda.is_available(), "PyTorch should be CPU build as configured"
    x = torch.ones(2, 2)
    y = x + x
    assert y.shape == (2, 2)

def test_transformers_imports():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    assert AutoTokenizer is not None
    assert AutoModelForSeq2SeqLM is not None
    assert pipeline is not None

def test_evaluation_metrics():
    import rouge_score
    import sacrebleu
    import evaluate
    assert rouge_score is not None
    assert sacrebleu is not None
    assert evaluate is not None
