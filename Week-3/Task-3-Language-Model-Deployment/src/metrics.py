"""
NEXUS NLP — Standardized Evaluation Metrics Suite
Implements ROUGE, SacreBLEU, Exact Match, Token F1, and Classification Accuracy.
Author: Subham Gupta (ShadowFox ML Engineer Intern)
Week 3: Task 3 - Language Model Deployment & Intelligence Evaluation Lab
"""

import re
import string
from collections import Counter
from typing import List, Dict, Any, Union
import numpy as np
import sacrebleu
from rouge_score import rouge_scorer


# ==============================================================================
# Text Normalization Utilities (Standard SQuAD / QA Protocol)
# ==============================================================================

def normalize_text(text: str) -> str:
    """
    Standardize text for evaluation:
    1. Lowercase
    2. Remove punctuation
    3. Remove articles ('a', 'an', 'the')
    4. Normalize whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()

    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # Remove punctuation
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)

    # Normalize whitespace
    text = " ".join(text.split())
    return text.strip()


# ==============================================================================
# Question Answering Metrics: Exact Match & Token-Level F1
# ==============================================================================

def compute_exact_match(prediction: str, reference: str) -> float:
    """Compute Exact Match (1.0 or 0.0) between normalized strings."""
    return float(normalize_text(prediction) == normalize_text(reference))


def compute_token_f1(prediction: str, reference: str) -> float:
    """
    Compute token-level precision, recall, and F1 score between normalized strings.
    """
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return float(pred_tokens == ref_tokens)

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(ref_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return float(f1)


def evaluate_qa(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute mean Exact Match and Token F1 across a batch of QA pairs."""
    if len(predictions) != len(references):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(references)} references")
    if len(predictions) == 0:
        return {"exact_match": 0.0, "f1": 0.0, "count": 0}

    ems = [compute_exact_match(p, r) for p, r in zip(predictions, references)]
    f1s = [compute_token_f1(p, r) for p, r in zip(predictions, references)]

    return {
        "exact_match": round(float(np.mean(ems)) * 100, 2),
        "f1": round(float(np.mean(f1s)) * 100, 2),
        "count": len(predictions)
    }


# ==============================================================================
# Summarization Metrics: ROUGE-1, ROUGE-2, ROUGE-L
# ==============================================================================

def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute standard ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.
    """
    if len(predictions) != len(references):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(references)} references")
    if len(predictions) == 0:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "count": 0}

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r1_scores, r2_scores, rl_scores = [], [], []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1_scores.append(scores['rouge1'].fmeasure)
        r2_scores.append(scores['rouge2'].fmeasure)
        rl_scores.append(scores['rougeL'].fmeasure)

    return {
        "rouge1": round(float(np.mean(r1_scores)) * 100, 2),
        "rouge2": round(float(np.mean(r2_scores)) * 100, 2),
        "rougeL": round(float(np.mean(rl_scores)) * 100, 2),
        "count": len(predictions)
    }


# ==============================================================================
# Translation Metrics: SacreBLEU
# ==============================================================================

def compute_sacrebleu(predictions: List[str], references: Union[List[str], List[List[str]]]) -> Dict[str, Any]:
    """
    Compute corpus-level SacreBLEU score against reference translations.
    """
    if len(predictions) == 0:
        return {"bleu": 0.0, "count": 0}

    # Handle 1D list of references by wrapping into list of lists (SacreBLEU format)
    if isinstance(references[0], str):
        refs = [[r] for r in references]
    else:
        refs = references

    # SacreBLEU expects references transposed as list of reference streams
    num_refs = len(refs[0])
    transposed_refs = [[refs[i][j] for i in range(len(refs))] for j in range(num_refs)]

    bleu = sacrebleu.corpus_bleu(predictions, transposed_refs)
    return {
        "bleu": round(float(bleu.score), 2),
        "precisions": [round(p, 2) for p in bleu.precisions],
        "bp": round(float(bleu.bp), 4),
        "sys_len": bleu.sys_len,
        "ref_len": bleu.ref_len,
        "count": len(predictions)
    }


# ==============================================================================
# Classification Metrics: Accuracy & Macro F1
# ==============================================================================

def compute_classification_metrics(predictions: List[str], references: List[str]) -> Dict[str, Any]:
    """
    Compute Classification Accuracy and detailed match counts.
    Normalizes string labels before comparison.
    """
    if len(predictions) != len(references):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(references)} references")
    if len(predictions) == 0:
        return {"accuracy": 0.0, "count": 0, "correct": 0}

    norm_preds = [normalize_text(p) for p in predictions]
    norm_refs = [normalize_text(r) for r in references]

    correct = sum(1 for p, r in zip(norm_preds, norm_refs) if p == r or r in p or p in r)
    accuracy = correct / len(predictions)

    return {
        "accuracy": round(float(accuracy) * 100, 2),
        "correct": correct,
        "total": len(predictions)
    }
