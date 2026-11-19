"""
NEXUS NLP — Standardized Evaluation Metrics Suite
Implements ROUGE, SacreBLEU, Exact Match, Token F1, and Classification Accuracy.
Author: Subham Gupta (ShadowFox ML Engineer Intern)
Task 3 - Language Model Deployment & Intelligence Evaluation Lab
"""

import re
import string
from collections import Counter
from typing import List, Dict, Any, Union, Optional
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
# Classification Metrics: Exact Canonical Matching & Macro F1
# ==============================================================================

def normalize_label(text: str) -> str:
    """
    Standardize a classification label:
    1. Lowercase
    2. Strip leading/trailing punctuation, quotes, and whitespace.
    """
    if not isinstance(text, str):
        return ""
    t = text.strip().lower()
    t = t.strip(string.punctuation + " \t\r\n")
    return t


def extract_predicted_label(
    prediction: str,
    allowed_labels: Optional[List[str]] = None
) -> Optional[str]:
    """
    Extract a canonical classification label from model prediction.
    Strict parsing rules:
    1. If allowed_labels is None, returns normalize_label(prediction).
    2. If normalize_label(prediction) exactly matches an allowed label, returns that label.
    3. If prediction contains extra text:
       - Identifies candidate matches as discrete words using word boundaries (\\b).
       - Explicitly rejects matches preceded by negation modifiers ('not', 'no', 'never', 'neither', 'non').
         E.g. 'not positive' must NOT match 'positive'.
       - Explicitly distinguishes prefixes like 'informal' from 'formal'.
       - If exactly one allowed label matches without negation, returns that canonical label.
       - If multiple conflicting allowed labels match (e.g. 'positive and negative'), returns None (ambiguous).
       - If zero allowed labels match, returns None.
    """
    norm_pred = normalize_label(prediction)
    if not norm_pred:
        return None

    if allowed_labels is None:
        return norm_pred

    norm_allowed_map = {normalize_label(label): normalize_label(label) for label in allowed_labels if normalize_label(label)}

    # Direct exact normalized match
    if norm_pred in norm_allowed_map:
        return norm_allowed_map[norm_pred]

    # Search for discrete allowed labels as whole words in lowercased text
    text_lower = prediction.lower()
    candidates = []

    for label_key in norm_allowed_map:
        # Match discrete word boundary
        pattern = rf"\b{re.escape(label_key)}\b"
        match = re.search(pattern, text_lower)
        if match:
            # Check for preceding negation within 20 chars before the match
            start_pos = match.start()
            prefix_window = text_lower[max(0, start_pos - 25):start_pos]
            is_negated = bool(re.search(r"\b(not|no|never|neither|non[\s\-])\s*$", prefix_window.strip()))
            if not is_negated:
                candidates.append(label_key)

    if len(candidates) == 1:
        return norm_allowed_map[candidates[0]]

    # Ambiguous (multiple candidates) or unmapped
    return None


def compute_classification_metrics(
    predictions: List[str],
    references: List[str],
    allowed_labels: Optional[Union[List[str], List[List[str]]]] = None
) -> Dict[str, Any]:
    """
    Compute rigorous classification metrics:
    - Accuracy: proportion of exact canonical label matches.
    - Macro F1: unweighted average of F1 scores across distinct reference classes.
    - Macro Precision and Macro Recall.
    - Counts: correct, total, unmatched predictions.

    Eliminates unsafe substring containment (e.g. 'formal' matching inside 'informal',
    or 'positive' matching inside 'not positive').
    """
    if len(predictions) != len(references):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(references)} references")
    if len(predictions) == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "correct": 0,
            "total": 0,
            "unmatched": 0
        }

    canonical_preds = []
    canonical_refs = [normalize_label(r) for r in references]

    for i, p in enumerate(predictions):
        item_allowed = None
        if allowed_labels is not None:
            if len(allowed_labels) > 0 and isinstance(allowed_labels[0], list):
                item_allowed = allowed_labels[i]
            else:
                item_allowed = allowed_labels

        extracted = extract_predicted_label(p, item_allowed)
        if extracted is not None:
            canonical_preds.append(extracted)
        else:
            canonical_preds.append("__unmatched__")

    correct = sum(1 for p, r in zip(canonical_preds, canonical_refs) if p == r)
    accuracy = correct / len(predictions) if predictions else 0.0
    unmatched_count = sum(1 for p in canonical_preds if p == "__unmatched__")

    # Compute Macro F1, Precision, and Recall using scikit-learn
    try:
        from sklearn.metrics import precision_recall_fscore_support
        unique_classes = sorted(list(set(canonical_refs)))
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            canonical_refs,
            canonical_preds,
            labels=unique_classes,
            average="macro",
            zero_division=0
        )
    except Exception:
        p_macro, r_macro, f1_macro = accuracy, accuracy, accuracy

    return {
        "accuracy": round(float(accuracy) * 100, 2),
        "macro_f1": round(float(f1_macro) * 100, 2),
        "macro_precision": round(float(p_macro) * 100, 2),
        "macro_recall": round(float(r_macro) * 100, 2),
        "correct": correct,
        "total": len(predictions),
        "unmatched": unmatched_count
    }
