"""
NEXUS NLP — Tab C: Decoding Strategy Comparator Component
Evaluates the same prompt simultaneously across multiple decoding strategies
with live, non-cached wall-clock timing and side-by-side output visualization.

Author: Subham Gupta (ShadowFox ML Engineer Intern)
"""

import streamlit as st
from src.model_engine import NexusModelEngine

COMPARATOR_PRESETS = [
    "Write an intuitive analogy explaining why neural networks require multiple layers of non-linear activations:",
    "Summarize: The James Webb Space Telescope was launched in December 2021 as NASA's premier space observatory. Operating at the Second Sun-Earth Lagrange Point 1.5 million kilometers away, it uses ultra-sensitive infrared sensors and an origami-style gold-coated beryllium mirror to observe the earliest stars and galaxies formed after the Big Bang.",
    "Translate English to French: Artificial intelligence is transforming modern science and healthcare.",
    "Explain quantum computing to a high school student in three concise sentences:"
]


def render_tab_comparator(engine: NexusModelEngine):
    """Renders Section C: Decoding Strategy Comparator."""
    st.subheader("Decoding Strategy Comparator")
    st.markdown(
        "Evaluate identical prompts under different search and sampling regimes side-by-side. "
        "Every generation is timed **live on CPU**—no pre-calculated or cached metrics are used."
    )

    preset_choice = st.selectbox(
        "Choose a Comparison Prompt Preset or Type Custom:",
        ["Custom Prompt"] + COMPARATOR_PRESETS
    )

    if preset_choice != "Custom Prompt":
        prompt_val = preset_choice
    else:
        prompt_val = "Write a creative analogy explaining how an artificial neural network processes information:"

    comp_prompt = st.text_area(
        "Comparison Prompt:",
        value=prompt_val,
        height=100
    )

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)

    with col_cfg1:
        st.markdown("##### Configuration 1 (Deterministic Baseline)")
        strat1 = st.selectbox("Strategy 1:", ["greedy", "beam_search"], index=0, key="strat1")
        beams1 = 1
        if strat1 == "beam_search":
            beams1 = st.slider("Beam Width (Cfg 1):", 2, 8, 4, key="beams1")
        tokens1 = st.slider("Max Tokens (Cfg 1):", 16, 128, 48, step=8, key="tok1")
        rep1 = st.slider("Repetition Penalty (Cfg 1):", 1.0, 2.5, 1.0, 0.1, key="rep1")

    with col_cfg2:
        st.markdown("##### Configuration 2 (Beam Search)")
        strat2 = st.selectbox("Strategy 2:", ["beam_search", "greedy"], index=0, key="strat2")
        beams2 = 4
        if strat2 == "beam_search":
            beams2 = st.slider("Beam Width (Cfg 2):", 2, 8, 4, key="beams2")
        tokens2 = st.slider("Max Tokens (Cfg 2):", 16, 128, 48, step=8, key="tok2")
        rep2 = st.slider("Repetition Penalty (Cfg 2):", 1.0, 2.5, 1.0, 0.1, key="rep2")

    with col_cfg3:
        st.markdown("##### Configuration 3 (Stochastic Sampling)")
        strat3 = "sampling"
        st.info("Strategy 3: Nucleus Sampling")
        temp3 = st.slider("Temperature (Cfg 3):", 0.1, 2.0, 0.7, 0.05, key="temp3")
        topp3 = st.slider("Top-p Nucleus (Cfg 3):", 0.1, 1.0, 0.9, 0.05, key="topp3")
        tokens3 = st.slider("Max Tokens (Cfg 3):", 16, 128, 48, step=8, key="tok3")
        rep3 = st.slider("Repetition Penalty (Cfg 3):", 1.0, 2.5, 1.0, 0.1, key="rep3")

    if st.button("⚔️ Execute Live Side-by-Side Comparison", type="primary", width="stretch"):
        if not comp_prompt.strip():
            st.error("Please enter a non-empty prompt.")
            return

        with st.spinner("Executing live multi-strategy inference on CPU ..."):
            # Config 1
            res1 = engine.generate(
                prompt=comp_prompt,
                strategy=strat1,
                num_beams=beams1,
                max_new_tokens=tokens1,
                repetition_penalty=rep1
            )

            # Config 2
            res2 = engine.generate(
                prompt=comp_prompt,
                strategy=strat2,
                num_beams=beams2,
                max_new_tokens=tokens2,
                repetition_penalty=rep2
            )

            # Config 3
            res3 = engine.generate(
                prompt=comp_prompt,
                strategy="sampling",
                temperature=temp3,
                top_p=topp3,
                top_k=50,
                max_new_tokens=tokens3,
                repetition_penalty=rep3
            )

        st.markdown("---")
        st.markdown("#### Live Side-by-Side Results:")

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.markdown(f"**Strategy 1: {strat1.upper()}**")
            st.caption(f"Beams: {beams1} | Rep. Pen: {rep1}")
            st.markdown(f"""
            <div style="background: #f8fafc; padding: 0.9rem; border: 1px solid #cbd5e1; border-radius: 6px; min-height: 120px; font-size: 0.95rem;">
                {res1.generated_text if res1.generated_text else "<em>[Empty generation]</em>"}
            </div>
            """, unsafe_allow_html=True)
            st.write("")
            st.metric("Live Latency", f"{res1.latency_ms:.1f} ms")
            st.metric("Throughput", f"{res1.tokens_per_second:.1f} tok/s")
            st.metric("Output Tokens", f"{res1.output_tokens} tok")

        with res_col2:
            st.markdown(f"**Strategy 2: {strat2.upper()}**")
            st.caption(f"Beams: {beams2} | Rep. Pen: {rep2}")
            st.markdown(f"""
            <div style="background: #f8fafc; padding: 0.9rem; border: 1px solid #cbd5e1; border-radius: 6px; min-height: 120px; font-size: 0.95rem;">
                {res2.generated_text if res2.generated_text else "<em>[Empty generation]</em>"}
            </div>
            """, unsafe_allow_html=True)
            st.write("")
            st.metric("Live Latency", f"{res2.latency_ms:.1f} ms")
            st.metric("Throughput", f"{res2.tokens_per_second:.1f} tok/s")
            st.metric("Output Tokens", f"{res2.output_tokens} tok")

        with res_col3:
            st.markdown(f"**Strategy 3: SAMPLING**")
            st.caption(f"Temp: {temp3} | Top-p: {topp3} | Rep. Pen: {rep3}")
            st.markdown(f"""
            <div style="background: #f8fafc; padding: 0.9rem; border: 1px solid #cbd5e1; border-radius: 6px; min-height: 120px; font-size: 0.95rem;">
                {res3.generated_text if res3.generated_text else "<em>[Empty generation]</em>"}
            </div>
            """, unsafe_allow_html=True)
            st.write("")
            st.metric("Live Latency", f"{res3.latency_ms:.1f} ms")
            st.metric("Throughput", f"{res3.tokens_per_second:.1f} tok/s")
            st.metric("Output Tokens", f"{res3.output_tokens} tok")
