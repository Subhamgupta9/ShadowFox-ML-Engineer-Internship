"""
NEXUS NLP — Tab D: Model Architecture & Safety Inspector Component
Presents verified structural specifications of FLAN-T5-small, detailed failure
analysis cases, hallucination vulnerabilities, and ethical deployment guardrails.

Author: Subham Gupta (ShadowFox ML Engineer Intern)
"""

import streamlit as st
from src.model_engine import NexusModelEngine


def render_tab_inspector(engine: NexusModelEngine):
    """Renders Section D: Model Architecture & Safety Inspector."""
    st.subheader("Model Architecture & Safety Inspector")
    st.markdown(
        "Inspect the verified architectural topology of `google/flan-t5-small`, study concrete failure "
        "case studies identified during Phase 3 benchmarking, and review production deployment guardrails."
    )

    specs = engine.get_specs()
    cfg = specs.get("config", {})

    # Critical Model Identity Callout
    st.info(
        "🧠 **Model Classification & Scale Awareness:**\n\n"
        "`google/flan-t5-small` is a **compact foundation model containing ~76.96 million parameters**. "
        "It is fundamentally designed for fast, low-latency, edge and CPU deployment across structured text-to-text tasks. "
        "**It is NOT a frontier-scale reasoning model** (such as Gemini 1.5 Pro, Claude 3.5 Sonnet, or GPT-4). "
        "It possesses significant architectural and parametric bounds regarding multi-step mathematical reasoning, "
        "long-context integration, and factual recall."
    )

    # Architectural Specifications Table / Cards
    st.markdown("#### Verified Architectural Specifications (from `model.config`):")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Parameters", f"{cfg.get('total_parameters', 76961152):,}")
    c2.metric("Encoder / Decoder Layers", f"{cfg.get('num_layers', 8)} / {cfg.get('num_decoder_layers', 8)}")
    c3.metric("Attention Heads", f"{cfg.get('num_heads', 6)} heads")
    c4.metric("Hidden Size (d_model)", f"{cfg.get('d_model', 512)} dim")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Feed-Forward (d_ff)", f"{cfg.get('d_ff', 1024)} dim")
    c6.metric("Feed-Forward Activation", f"{cfg.get('feed_forward_proj', 'gated-gelu')}")
    c7.metric("Vocabulary Size", f"{cfg.get('vocab_size', 32128):,} pieces")
    c8.metric("Positional Encoding", "Relative Bias (32 buckets)")

    st.markdown("---")

    # Failure Mode Case Studies (Observed in Phase 3)
    st.markdown("#### Concrete Failure Analysis & Empirical Error Modes:")
    st.markdown(
        "During our Phase 3 empirical evaluation, four major failure archetypes were identified and cataloged:"
    )

    f1, f2 = st.columns(2)

    with f1:
        with st.expander("❌ Case 1: Multi-Step Arithmetic Failure", expanded=True):
            st.markdown("""
            - **Input Prompt:** `Solve this arithmetic problem: If a factory produces 25 units per hour and operates for 8 hours a day, how many units does it produce in 5 days? Answer with the final number:`
            - **Expected Correct Answer:** `1000` ($25 \\times 8 \\times 5 = 1000$)
            - **Actual FLAN-T5-small Prediction:** `84`
            - **Root Cause Analysis:** Encoder-decoder language models do not execute symbolic algebraic calculations internally. Without external calculator tools or chain-of-thought (CoT) scratchpads, sub-100M parameter models match superficial statistical token transitions rather than computing arithmetic.
            """)

        with f2:
            with st.expander("❌ Case 2: Riddle / Abductive Association Hallucination", expanded=True):
                st.markdown("""
                - **Input Prompt:** `Solve this riddle: I speak without a mouth and hear without ears. I have no body, but I come alive with wind. What am I?`
                - **Expected Ground Truth:** `echo`
                - **Actual FLAN-T5-small Prediction:** `I am a sailor`
                - **Root Cause Analysis:** The model attended disproportionately to the token "wind" and generated a strong maritime statistical prior ("sailor") rather than performing semantic constraint satisfaction over the full metaphorical riddle.
                """)

    f3, f4 = st.columns(2)

    with f3:
        with st.expander("⚠️ Case 3: Tone & Register Classification Drift"):
            st.markdown("""
            - **Input Prompt:** `Classify the tone of this statement as Formal or Informal: We are writing to formally acknowledge receipt of your correspondence dated March 14th.`
            - **Expected Ground Truth:** `Formal`
            - **Actual FLAN-T5-small Prediction:** `Informal`
            - **Root Cause Analysis:** Surface sensitivity in zero-shot classification without explicit in-context few-shot exemplars defining bureaucratic linguistic boundaries.
            """)

    with f4:
        with st.expander("⚠️ Case 4: Literalism in Multi-Clause Translation"):
            st.markdown("""
            - **Input Prompt:** `Translate English to Spanish: Machine learning models require clean, well-curated training data.`
            - **Reference Translation:** `Los modelos de aprendizaje automático requieren datos de entrenamiento limpios y bien seleccionados.`
            - **Actual Prediction:** `Modelos de aprendizaje de máquinas requiere datos limpias y bien curated.`
            - **Root Cause Analysis:** Retains loanwords ("curated") and exhibits minor gender agreement inflection errors ("datos limpias" instead of "datos limpios").
            """)

    st.markdown("---")

    # Safety & Ethical Considerations
    st.markdown("#### Ethical Considerations & Production Deployment Guardrails:")

    g1, g2, g3 = st.columns(3)

    with g1:
        st.markdown("##### 1. Pre-Training Corpus Bias")
        st.write(
            "FLAN-T5 was pre-trained on Google's Colossal Clean Crawled Corpus (C4). "
            "Web data inherently incorporates demographic skews, cultural biases, and representational disparities. "
            "Output filters should audit for demographic stereotyping prior to user delivery."
        )

    with g2:
        st.markdown("##### 2. Hallucination Risk in High-Stakes Tasks")
        st.write(
            "Compact parameter budgets force aggressive semantic compression. When queried on specialized "
            "medical, legal, or financial topics outside its parametric memory, the model generates confident, fluent "
            "yet fabricated assertions. Never deploy ungrounded models in critical sectors."
        )

    with g3:
        st.markdown("##### 3. Recommended Production Guardrails")
        st.write(
            "1. **RAG Grounding:** Provide factual retrieval context in the prompt.\n"
            "2. **Strict Temperature Limits:** Keep $T \\le 0.7$ for factual tasks.\n"
            "3. **Length Caps:** Bound `max_new_tokens` to prevent runaway generation.\n"
            "4. **Human-in-the-Loop:** Require human verification for sensitive decisions."
        )
