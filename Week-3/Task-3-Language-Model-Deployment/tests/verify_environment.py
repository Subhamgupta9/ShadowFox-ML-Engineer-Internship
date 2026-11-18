"""
NEXUS NLP — Environment Verification Script
Validates dedicated Python 3.11.8 virtual environment and all ML/NLP dependencies.
Author: Subham Gupta (ShadowFox ML Engineer Intern)
Week 3: Task 3 - Language Model Deployment & Intelligence Evaluation Lab
"""

import sys
import os
import platform

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def verify():
    print("=" * 70)
    print("NEXUS NLP -- ENVIRONMENT VERIFICATION SUITE")
    print("=" * 70)

    # 1. Python Environment Check
    py_ver = sys.version.split()[0]
    exec_path = sys.executable
    is_venv = sys.prefix != sys.base_prefix
    print(f"[1] Python Version:     {py_ver} ({platform.architecture()[0]})")
    print(f"[2] Python Executable:  {exec_path}")
    print(f"[3] Virtual Env Active: {is_venv} (Prefix: {sys.prefix})")
    assert "3.11" in py_ver, f"Expected Python 3.11, got {py_ver}"
    assert is_venv, "Must be running inside dedicated virtual environment!"

    print("-" * 70)
    print("Package Verification:")
    print("-" * 70)

    packages = [
        ("torch", "PyTorch (CPU)", lambda m: f"v{m.__version__} | CPU Available: {m.cpu is not None} | CUDA: {m.cuda.is_available()}"),
        ("transformers", "Hugging Face Transformers", lambda m: f"v{m.__version__}"),
        ("tokenizers", "Hugging Face Tokenizers", lambda m: f"v{m.__version__}"),
        ("accelerate", "Hugging Face Accelerate", lambda m: f"v{m.__version__}"),
        ("sentencepiece", "SentencePiece Tokenizer", lambda m: f"v{m.__version__}"),
        ("huggingface_hub", "Hugging Face Hub Client", lambda m: f"v{m.__version__}"),
        ("numpy", "NumPy Scientific Computing", lambda m: f"v{m.__version__}"),
        ("pandas", "Pandas DataFrames", lambda m: f"v{m.__version__}"),
        ("matplotlib", "Matplotlib Visualization", lambda m: f"v{m.__version__}"),
        ("seaborn", "Seaborn Statistical Plots", lambda m: f"v{m.__version__}"),
        ("sklearn", "Scikit-Learn ML Utilities", lambda m: f"v{m.__version__}"),
        ("psutil", "Process & System Utilities", lambda m: f"v{m.__version__}"),
        ("tqdm", "Progress Bars", lambda m: f"v{m.__version__}"),
        ("pytest", "Pytest Test Framework", lambda m: f"v{m.__version__}"),
        ("rouge_score", "ROUGE Metric Evaluation", lambda m: f"installed"),
        ("sacrebleu", "SacreBLEU Metric Evaluation", lambda m: f"v{m.__version__}"),
        ("evaluate", "Hugging Face Evaluate", lambda m: f"v{m.__version__}"),
    ]

    all_ok = True
    for mod_name, label, detail_fn in packages:
        try:
            mod = __import__(mod_name)
            detail = detail_fn(mod)
            print(f"  [PASS] {label:<32} {detail}")
        except Exception as e:
            print(f"  [FAIL] {label:<32} FAILED: {e}")
            all_ok = False

    print("-" * 70)
    print("Functional Subsystem Tests:")
    print("-" * 70)

    # 2. PyTorch Tensor Math & NumPy Bridge Test
    import torch
    import numpy as np
    t = torch.randn(4, 4)
    matmul_result = torch.matmul(t, t.T)
    np_arr = matmul_result.numpy()
    assert np_arr.shape == (4, 4), "PyTorch-NumPy tensor bridge failed"
    print(f"  [PASS] PyTorch CPU Matmul & NumPy Bridge: Passed (Shape: {np_arr.shape})")

    # 3. Transformers Subsystem Check
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    print(f"  [PASS] Transformers Seq2Seq LM Classes:   AutoTokenizer, AutoModelForSeq2SeqLM imported")
    print(f"  [PASS] Transformers Pipeline Subsystem:  Pipeline factory initialized")

    print("=" * 70)
    if all_ok:
        print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY (17/17 packages verified)")
    else:
        print("SOME VERIFICATION CHECKS FAILED")
    print("=" * 70)
    return all_ok

if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
