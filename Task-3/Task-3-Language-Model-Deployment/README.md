# NEXUS NLP: Language Model Deployment & Intelligence Evaluation Lab
### Week 3 — Advanced Task 3 | ShadowFox Machine Learning Engineer Internship
**Author:** Subham Gupta ([@Subhamgupta9](https://github.com/Subhamgupta9))
**Model Architecture:** Google FLAN-T5-Small (`google/flan-t5-small`, ~76.96 Million Parameters)
**Execution Environment:** Isolated Python 3.11.8 Virtual Environment (`venv`), PyTorch 2.15.0+cpu, Hugging Face Transformers 5.23.0 on Intel x86_64 CPU
**Interactive Dashboard:** Streamlit Web Application (Port 8503)
**Jupyter Kernel:** `NEXUS-NLP-Python311` (`nexus-nlp-python311`)

---

## 1. Project Overview & Research Objectives

**NEXUS NLP** is an end-to-end Language Model Deployment & Multi-Task Intelligence Evaluation Laboratory engineered as part of the **ShadowFox Machine Learning Engineer Internship** (Week 3, Advanced Task 3).

The primary objective of this project is to implement, evaluate, analyze, and deploy a state-of-the-art Sequence-to-Sequence (Seq2Seq) Transformer foundation model:
- **Foundational Implementation:** Build a modular, CPU-optimized inference engine from scratch using Hugging Face Transformers.
- **Architectural Dissection:** Dissect FLAN-T5's encoder-decoder topology, relative position biases, Gated GELU feed-forward networks, and instruction tuning methodology (FLAN: Fine-tuned Language Net).
- **Multi-Task Benchmark:** Quantitatively evaluate the model across 5 canonical NLP capabilities:
  1. *Abstractive & Extractive Summarization* (ROUGE-1, ROUGE-2, ROUGE-L)
  2. *Context-Grounded Question Answering* (Exact Match, Token-level F1)
  3. *Zero-Shot Text Classification* (Accuracy across sentiment, topic, and tone)
  4. *Multi-Lingual Machine Translation* (SacreBLEU across French, German, Spanish)
  5. *Linguistic & Arithmetic Reasoning* (Controlled qualitative correctness)
- **Decoding Analysis:** Analyze the trade-offs of decoding strategies (Greedy Search, Beam Search, Temperature Scaling, Top-p Nucleus Sampling, Repetition Penalties).
- **Production CPU Profiling:** Measure wall-clock inference latency, tokens generated per second, and memory consumption.
- **Safety & Error Analysis:** Rigorously document model failure modes, arithmetic vulnerabilities, and ethical deployment guardrails.
- **Interactive Web Application:** Deploy a professional 4-tab Streamlit laboratory interface for live multi-task inference, benchmark browsing, side-by-side strategy comparison, and architectural safety inspection.

---

## 2. Verified Architectural Specifications (`google/flan-t5-small`)

The architectural parameters were inspected directly from the loaded `model.config` and official checkpoint:

| Architectural Attribute | Verified Specification | Description / Engineering Role |
| :--- | :--- | :--- |
| **Model ID** | `google/flan-t5-small` | Official checkpoint on Hugging Face Model Hub |
| **Model Type** | `t5` (Encoder-Decoder) | Bidirectional encoder with autoregressive causal decoder |
| **Total Parameters** | **76,961,152** (~76.96M) | 100% trainable floating-point parameters |
| **Hidden Size ($d_{model}$)** | **512** | Embedding and representation dimension |
| **Key/Value Dimension ($d_{kv}$)** | **64** | Dimension per attention head |
| **Attention Heads** | **6** | Multi-head attention projections |
| **Feed-Forward Dimension ($d_{ff}$)** | **1024** | Intermediate dimension of position-wise feedforward blocks |
| **Feed-Forward Activation** | **`gated-gelu`** | Gated GELU activation (T5v1.1 standard; verified in config) |
| **Encoder Layers** | **8** | Bidirectional transformer encoder blocks |
| **Decoder Layers** | **8** | Causal autoregressive transformer decoder blocks |
| **Positional Embeddings** | **Relative Position Bias** | 32 logarithmic distance buckets (max distance 128) |
| **Normalization** | **RMSNorm** | Root Mean Square Layer Normalization without learned bias |
| **Vocabulary Size** | **32,128** | SentencePiece unigram subword vocabulary |
| **Storage Footprint** | **~308 MB** (`model.safetensors`) | Compact size ideal for CPU and edge deployment |
| **RAM Footprint Delta** | **+343.91 MB** | Resident Set Size (RSS) memory delta on CPU load |

---

## 3. Preliminary Multi-Task Benchmark Results & Evaluation Scope

> [!IMPORTANT]
> **Preliminary Benchmark Scope & Sample Size Limitations:**
> The quantitative metrics presented below are derived from a preliminary evaluation benchmark suite ($N=4$ to $6$ curated exemplars per task) designed to validate end-to-end inference mechanics, test metric computation pipelines, and verify directional model behavior.
> **These scores must not be interpreted as universal, definitive, or production-grade performance metrics.** Production-grade assessment requires running large-scale standardized datasets (e.g., SQuAD v2, CNN/DailyMail test split, WMT-16, SuperGLUE) across thousands of test instances. Actual measured scores on our curated benchmark suite are documented transparently below without modification.

All evaluations were executed on CPU using the standardized evaluation harness ([src/benchmark.py](src/benchmark.py)) and recorded in [reports/benchmark_results.csv](reports/benchmark_results.csv):

| Task Domain | Evaluation Metric | Score | Sample Size ($N$) | Recommended Strategy | Empirical Finding |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Summarization** | ROUGE-L F1 | **21.72%** | 4 articles (Preliminary) | Beam Search ($k=4$) | Extracts core topics reliably; conservative length. |
| **Summarization** | ROUGE-1 / ROUGE-2 | **23.44% / 10.47%** | 4 articles (Preliminary) | Beam Search ($k=4$) | Captures primary key phrases; strong unigram overlap. |
| **Question Answering** | Token-Level F1 | **80.00%** | 5 contexts (Preliminary) | Greedy Search | High precision on factual, context-grounded queries. |
| **Question Answering** | Exact Match (EM) | **80.00%** | 5 contexts (Preliminary) | Greedy Search | Matches ground-truth spans with SQuAD normalization. |
| **Zero-Shot Classification**| Accuracy / Macro-F1 | **66.67% / 66.67%** | 6 texts (Preliminary) | Greedy Search | Strict exact-label evaluation (correcting preliminary substring inflation). |
| **Machine Translation** | SacreBLEU | **19.84** | 5 sentences (Preliminary) | Beam Search ($k=4$) | High lexical alignment; minor syntax errors in long clauses. |
| **Reasoning & Logic** | Correctness | **25.00%** | 4 problems (Preliminary) | Greedy Search | Solves deductive syllogisms; fails multi-step arithmetic. |


> [!NOTE]
> **Methodological Separation:** Automatic surface metrics (BLEU, ROUGE) are **not** universal measures of reasoning quality. Surface string overlap cannot assess arithmetic correctness. Factual reasoning was evaluated against exact computational solutions, exposing model limitations without fabricating capability.

---

## 4. Decoding Strategy & CPU Performance Analysis

### Strategy Comparison on Identical Prompts
| Strategy Configuration | Decoding Mode | Latency (ms) | Throughput (tok/s) | Output Quality Profile |
| :--- | :--- | :---: | :---: | :--- |
| **Greedy Search** | Deterministic ($k=1$) | **~135 – 250 ms** | **22.5 tok/s** | Highest speed, low repetition, optimal for QA & Classification. |
| **Beam Search ($k=2$)** | Deterministic ($B=2$) | **~480 ms** | **18.2 tok/s** | Improved coherence over greedy; balanced latency. |
| **Beam Search ($k=4$)** | Deterministic ($B=4$) | **~850 – 1200 ms** | **16.5 tok/s** | Maximum syntactic fluency; optimal for Summarization & Translation. |
| **Nucleus Sampling ($T=0.7, p=0.9$)** | Stochastic | **~380 ms** | **20.8 tok/s** | Natural diversity; avoids degeneration; best for open-ended text. |
| **High-Temp Sampling ($T=1.2, p=0.95$)**| Stochastic | **~410 ms** | **20.1 tok/s** | High creativity; higher risk of hallucination or drift. |
| **Repetition Penalty ($2.0$)** | Deterministic | **~240 ms** | **21.9 tok/s** | Strongly prevents cyclic n-gram loops. |

### CPU Hardware Performance Telemetry
- **Mean Generation Latency:** ~150 ms (short outputs) to ~1200 ms (long sequences).
- **Throughput Distribution:** Centered at **21.2 tokens/second** on standard 8-core CPU.
- **Scaling Complexity:** Linear $O(N)$ autoregressive scaling ($\approx 42\text{ ms}$ per generated token).
- **Peak RAM Footprint:** Stable at **~706 MB** total process memory (zero memory leaks observed).

---

## 5. Interactive Streamlit Application (Port 8503)

The project includes an interactive web dashboard built with Streamlit and Plotly, organized into four specialized tabs:

### Tab 1: Interactive NLP Lab
- **Task Presets & Custom Input:** One-click presets for Summarization, Context-based QA, Zero-Shot Classification, Translation, Instruction Following, and Reasoning.
- **Decoding Controls:** Real-time toggles for Greedy Search, Beam Width ($k=2$ to $8$), Temperature, Top-p Nucleus, Top-k, and Repetition Penalty.
- **Live Performance Telemetry:** Displays actual wall-clock generation latency (ms), output tokens, generation throughput (tok/s), input tokens, and current process RSS RAM.
- **Output Transparency:** Transparent warning reminding users that generated text can contain hallucinations or arithmetic errors.

### Tab 2: Multi-Task Benchmark Dashboard
- **Scorecards with Sample Sizes:** Interactive metric cards displaying ROUGE-L ($N=4$), QA Token F1 ($N=5$), Classification Accuracy ($N=6$), Translation SacreBLEU ($N=5$), and Reasoning Correctness ($N=4$).
- **Preliminary Scope Callout:** Prominent disclaimer clarifying that preliminary sample sizes demonstrate directional capability, not universal performance.
- **Interactive Plotly Visualizations:** Dynamic bar charts of task performance and latency vs. throughput trade-offs.
- **Granular Trials Explorer:** Filterable and searchable table of all 45 recorded trials from `reports/benchmark_results.csv`.

### Tab 3: Decoding Strategy Comparator
- **Side-by-Side Multi-Strategy Inference:** Evaluates the exact same prompt across Greedy Search, Beam Search ($k=4$), and Nucleus Sampling ($T=0.7, p=0.9$) simultaneously.
- **Live Measured Performance:** Displays actual non-cached wall-clock latencies and throughputs measured live on CPU.
- **Side-by-Side Outputs:** Compare generated text, structural phrasing, and repetition characteristics across strategies.

### Tab 4: Model Architecture & Safety Inspector
- **Verified Specifications:** Live cards displaying 76.96M parameters, 8 encoder/decoder layers, 6 attention heads, `gated-gelu` activation, SentencePiece 32,128 tokens, and relative positional bias.
- **Scale Awareness Notice:** Explicitly labels the model as a compact 77M foundation model, not a frontier-scale reasoning engine.
- **Empirical Failure Case Studies:** Catalogs real errors discovered during evaluation:
  1. *Multi-step arithmetic failure:* $25 \times 8 \times 5 = 1000$ predicted as $84$.
  2. *Metaphorical riddle hallucination:* Wind riddle predicted as "I am a sailor" instead of "echo".
  3. *Tone classification drift:* Bureaucratic formal statement misclassified as informal.
  4. *Translation loanword literalism:* Minor loanwords and gender agreement artifacts.
- **Deployment Guardrails:** Recommendations for RAG grounding, temperature limits, length caps, and human-in-the-loop oversight.

---

## 6. Directory Layout

```text
Task-3/Task-3-Language-Model-Deployment/
├── venv/                                # [Ignored by git] Dedicated Python 3.11.8 Virtual Environment
├── model_cache/                         # [Ignored by git] Downloaded FLAN-T5-small checkpoint
├── app/                                 # Modular Streamlit Web Application
│   ├── __init__.py
│   ├── app.py                           # Main application entrypoint & singleton cache
│   └── components/                      # Modular UI tab components
│       ├── __init__.py
│       ├── ui_header.py                 # Hero header, badges, and advisory notices
│       ├── tab_lab.py                   # Tab 1: Interactive NLP Lab & live inference
│       ├── tab_benchmark.py             # Tab 2: Multi-Task Benchmark Dashboard
│       ├── tab_comparator.py            # Tab 3: Decoding Strategy Comparator
│       └── tab_inspector.py             # Tab 4: Architecture & Safety Inspector
├── notebooks/                           # Executed research notebook
│   └── nexus_nlp_lab.ipynb              # Fully executed 19-section research notebook with all outputs
├── reports/                             # Evaluation results & visualizations
│   ├── figures/
│   │   ├── screenshots/                 # Application browser QA screenshots
│   │   │   ├── tab1_interactive_lab.png
│   │   │   ├── tab2_benchmark_dashboard.png
│   │   │   ├── tab3_decoding_comparator.png
│   │   │   └── tab4_architecture_safety.png
│   │   ├── task_performance_summary.png
│   │   ├── decoding_strategy_comparison.png
│   │   ├── latency_scaling_analysis.png
│   │   └── throughput_distribution.png
│   ├── benchmark_results.csv            # 45 granular inference evaluation records
│   ├── task_metrics_summary.json        # Unified task metrics JSON
│   └── model_architecture_spec.json     # Verified architectural parameters JSON
├── src/                                 # Modular Python inference engine & benchmark harness
│   ├── __init__.py
│   ├── model_engine.py                  # NexusModelEngine: Model lifecycle, tokenization, decoding
│   ├── metrics.py                       # Standardized ROUGE, SacreBLEU, EM, Token F1, Accuracy
│   └── benchmark.py                     # Multi-task benchmark & profiling runner
├── tests/                               # Automated unit test suite (30/30 passing)
│   ├── __init__.py
│   ├── test_environment.py             # Environment & PyTorch CPU tests
│   ├── test_model_engine.py             # Engine, validation, decoding & metric tests
│   ├── test_metrics.py                  # Strict classification & NLP metric tests
│   ├── test_app_components.py          # Streamlit UI component tests
│   └── verify_environment.py            # Environment diagnostic utility
├── requirements.txt                     # Pinned reproducible dependencies
└── README.md                            # Comprehensive project documentation
```

---

## 7. Reproduction & Execution Guide

### Step 1: Environment Setup & Activation

**Windows (PowerShell) — Verified on Python 3.11.8 Windows x86_64:**
```powershell
cd Task-3\Task-3-Language-Model-Deployment
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install "torch==2.15.0+cpu" --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
*(Note: Quoting `"torch==2.15.0+cpu"` in PowerShell prevents operator parsing errors).*

**Linux / macOS (Generic Reference — Not Locally Verified in this Environment):**
```bash
cd Task-3/Task-3-Language-Model-Deployment
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Run Verification & Automated Tests
```powershell
python -m pytest -v tests/
```
*Expected: 30+ passed.*

### Step 3: Run Multi-Task Benchmark Harness

```powershell
python src/benchmark.py
```
*Generates reports/benchmark_results.csv and regenerates all figures in reports/figures/.*

### Step 4: Launch Interactive Streamlit Application
```powershell
streamlit run app/app.py --server.port 8503
```
*Open your browser at `http://localhost:8503`.*

### Step 5: Open or Re-execute Research Notebook
```powershell
jupyter notebook notebooks/nexus_nlp_lab.ipynb
```
*Select kernel: `NEXUS-NLP-Python311`.*

---

## 8. Known Issues, Limitations & Future Work

1. **Arithmetic Limitations:** Like most compact language models under 100M parameters, FLAN-T5-small cannot reliably perform multi-step arithmetic without external calculator APIs or chain-of-thought (CoT) fine-tuning.
2. **Context Window Constraint:** The maximum input sequence length is bounded to 512 tokens by default, requiring chunking strategies for long documents.
3. **CPU Latency on Long Sequences:** While short generations (16–32 tokens) complete in 150–250 ms, long beam-search sequences (64+ tokens) require ~1000–1200 ms on CPU.
4. **Future Quantization Roadmap (Theoretical Target):** Exporting the model to ONNX Runtime with INT8 post-training quantization is an estimated future research target (projected to theoretically reduce CPU latency by $\approx 2\text{--}2.5\times$ and lower working RAM to $\approx 150\text{ MB}$, though not yet measured in the current CPU PyTorch implementation).

---

## 9. Development Roadmap

- [x] **Phase 1:** Environment & Hardware Feasibility Audit *(Approved)*
- [x] **Phase 2:** Project Scaffolding & Environment Initialization *(Approved)*
- [x] **Phase 3:** Model Implementation, Multi-Task Benchmarking & Quantitative Evaluation *(Approved)*
- [x] **Phase 4:** Interactive Web Application Development & Deployment *(Completed)*
- [ ] **Phase 5:** Quality Assurance, GitHub Integration & Release
