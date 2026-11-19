"""
Model Download & Architectural Inspection Script for FLAN-T5-Small
Week 3: Task 3 - Language Model Deployment & Intelligence Evaluation Lab
Author: Subham Gupta (ShadowFox ML Engineer Intern)
"""

import sys
import os
import time
import json
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def run_inspection():
    print("=" * 75)
    print("NEXUS NLP -- FLAN-T5 MODEL DOWNLOAD & ARCHITECTURAL INSPECTION")
    print("=" * 75)

    process = psutil.Process(os.getpid())
    mem_initial_mb = process.memory_info().rss / (1024 * 1024)
    print(f"Initial Process RSS Memory: {mem_initial_mb:.2f} MB")

    model_id = "google/flan-t5-small"
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(base_dir, "model_cache")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Target Model Cache Directory: {cache_dir}")
    print(f"Downloading / Loading Model: {model_id} ...")

    # Measure Tokenizer loading
    t0_tok = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    tok_load_time = time.perf_counter() - t0_tok
    print(f"[PASS] Tokenizer loaded in {tok_load_time:.2f}s (Vocab size: {tokenizer.vocab_size})")

    # Measure Model loading
    t0_model = time.perf_counter()
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir=cache_dir)
    model.to("cpu")
    model.eval()
    model_load_time = time.perf_counter() - t0_model

    mem_loaded_mb = process.memory_info().rss / (1024 * 1024)
    mem_delta_mb = mem_loaded_mb - mem_initial_mb

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"[PASS] Model loaded in {model_load_time:.2f}s on CPU")
    print(f"Total Parameters:      {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable Parameters:  {trainable_params:,}")
    print(f"Process Memory After:  {mem_loaded_mb:.2f} MB (Delta: +{mem_delta_mb:.2f} MB)")

    # Inspect Configuration
    cfg = model.config
    print("-" * 75)
    print("VERIFIED ARCHITECTURAL SPECIFICATIONS (from model.config):")
    print("-" * 75)
    arch_details = {
        "model_type": getattr(cfg, "model_type", "t5"),
        "d_model (Hidden Size)": cfg.d_model,
        "d_kv (Key/Value Dim)": cfg.d_kv,
        "d_ff (Feed-Forward Dim)": cfg.d_ff,
        "num_layers (Encoder)": cfg.num_layers,
        "num_decoder_layers": cfg.num_decoder_layers,
        "num_heads": cfg.num_heads,
        "relative_attention_num_buckets": cfg.relative_attention_num_buckets,
        "relative_attention_max_distance": cfg.relative_attention_max_distance,
        "feed_forward_proj (Activation)": getattr(cfg, "feed_forward_proj", "relu"),
        "vocab_size": cfg.vocab_size,
        "is_encoder_decoder": cfg.is_encoder_decoder,
        "tie_word_embeddings": cfg.tie_word_embeddings,
        "initializer_factor": cfg.initializer_factor,
        "layer_norm_epsilon": cfg.layer_norm_epsilon,
    }
    for k, v in arch_details.items():
        print(f"  {k:<36}: {v}")

    print("-" * 75)
    print("INITIAL INFERENCE VERIFICATION (CPU):")
    print("-" * 75)
    test_prompts = [
        "Translate English to French: Good morning, welcome to our machine learning presentation.",
        "Q: What is the primary role of an attention mechanism in transformers? A:",
        "Summarize the following: Transfer learning allows machine learning models to leverage knowledge gained from pre-training on large datasets, reducing the training time and data required for downstream tasks.",
        "Classify the sentiment of this review as positive or negative: The new laptop is remarkably fast, silent, and has exceptional battery life."
    ]

    for prompt in test_prompts:
        t0 = time.perf_counter()
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=2,
                early_stopping=True
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        tokens_gen = outputs.shape[1]
        tps = (tokens_gen / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0

        print(f"Prompt:     {prompt[:70]}...")
        print(f"Generated:  {decoded}")
        print(f"Performance: {elapsed_ms:.1f} ms | Generated Tokens: {tokens_gen} | Speed: {tps:.1f} tok/s")
        print()

    # Save architectural metadata to json for reporting & notebook
    meta_path = os.path.join(base_dir, "reports", "model_architecture_spec.json")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_id": model_id,
            "total_params": total_params,
            "load_time_sec": round(model_load_time, 3),
            "memory_delta_mb": round(mem_delta_mb, 2),
            "arch_details": arch_details
        }, f, indent=2)
    print(f"Saved architectural specification to: {meta_path}")
    print("=" * 75)

if __name__ == "__main__":
    run_inspection()
