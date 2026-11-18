"""
NEXUS NLP — Tab A: Interactive NLP Lab Component
Provides task presets, custom text input, decoding controls, live inference,
and telemetry performance tracking.

Author: Subham Gupta (ShadowFox ML Engineer Intern)
"""

import streamlit as st
from src.model_engine import NexusModelEngine

TASK_PRESETS = {
    "Summarization": {
        "description": "Condense multi-sentence articles or technical passages into concise summaries.",
        "prompt": "Summarize the following article: Deep learning has revolutionized artificial intelligence over the past decade. By training deep neural networks on massive datasets with gradient-based optimization algorithms, models have achieved super-human capabilities in image classification, protein structure prediction, and natural language understanding. However, training these foundational architectures demands immense computational power, specialized tensor processing hardware, and substantial electrical energy consumption.",
        "recommended_strategy": "beam_search",
        "num_beams": 4,
        "max_new_tokens": 64
    },
    "Question Answering (Context-Grounded)": {
        "description": "Extract precise factual answers grounded within a supplied context.",
        "prompt": "Context: The Apollo 11 mission was the first spaceflight that landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969. Question: Who was the commander of the Apollo 11 mission?",
        "recommended_strategy": "greedy",
        "num_beams": 1,
        "max_new_tokens": 32
    },
    "Zero-Shot Classification": {
        "description": "Classify text into arbitrary categories (sentiment, topic, tone) zero-shot.",
        "prompt": "Classify the sentiment of this text as positive or negative: The customer service was astonishingly responsive and resolved my technical issue within minutes.",
        "recommended_strategy": "greedy",
        "num_beams": 1,
        "max_new_tokens": 16
    },
    "Machine Translation": {
        "description": "Translate natural language between English, French, German, and Spanish.",
        "prompt": "Translate English to French: Artificial intelligence is transforming modern science and healthcare.",
        "recommended_strategy": "beam_search",
        "num_beams": 4,
        "max_new_tokens": 48
    },
    "Instruction Following": {
        "description": "Follow direct structured instructions and transformation directives.",
        "prompt": "Convert the following statement into a polite professional email response: I cannot finish the report today because I am waiting for the sales numbers.",
        "recommended_strategy": "sampling",
        "num_beams": 1,
        "max_new_tokens": 64
    },
    "Reasoning-Style Prompts": {
        "description": "Evaluate multi-step logic and deductive syllogisms (highlights small-model reasoning boundaries).",
        "prompt": "Reason logically: All mammals are warm-blooded. All dolphins are mammals. Are dolphins warm-blooded? Answer yes or no:",
        "recommended_strategy": "greedy",
        "num_beams": 1,
        "max_new_tokens": 24
    }
}


def render_tab_lab(engine: NexusModelEngine):
    """Renders Section A: Interactive NLP Lab."""
    st.subheader("Interactive NLP Laboratory")
    st.markdown(
        "Explore real-time sequence-to-sequence inference with `google/flan-t5-small` executed strictly on local CPU. "
        "Select a task preset or provide your own custom prompt."
    )

    col_task, col_preset_btn = st.columns([3, 1])
    with col_task:
        selected_task = st.selectbox(
            "Select NLP Task:",
            list(TASK_PRESETS.keys()),
            help="Choose a pre-configured task category."
        )

    preset_data = TASK_PRESETS[selected_task]
    st.caption(f"**Task Role:** {preset_data['description']}")

    # Session state for prompt text
    if "lab_prompt" not in st.session_state:
        st.session_state.lab_prompt = preset_data["prompt"]

    with col_preset_btn:
        st.write("")
        st.write("")
        if st.button("Load Preset Prompt", width="stretch"):
            st.session_state.lab_prompt = preset_data["prompt"]
            st.rerun()

    # Input prompt
    user_prompt = st.text_area(
        "Input Prompt:",
        value=st.session_state.lab_prompt,
        height=140,
        help="Input text passed to the encoder. For best results, use standard task prefixes like 'Summarize:', 'Q: ... A:', or 'Translate English to French:'."
    )

    # Generation Controls
    with st.expander("⚙️ Generation Hyperparameters & Decoding Strategy", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            strategy = st.selectbox(
                "Decoding Strategy:",
                ["greedy", "beam_search", "sampling"],
                index=["greedy", "beam_search", "sampling"].index(preset_data.get("recommended_strategy", "greedy")),
                help="Greedy is fastest; Beam Search explores top paths; Sampling introduces stochastic diversity."
            )
            max_new_tokens = st.slider(
                "Max New Tokens:",
                min_value=8,
                max_value=256,
                value=preset_data.get("max_new_tokens", 64),
                step=8,
                help="Maximum number of tokens generated by the autoregressive decoder."
            )

        with col2:
            num_beams = 1
            if strategy == "beam_search":
                num_beams = st.slider(
                    "Beam Width (num_beams):",
                    min_value=2,
                    max_value=8,
                    value=preset_data.get("num_beams", 4),
                    help="Number of parallel beam paths maintained during search."
                )
            else:
                st.info(f"Deterministic mode ({strategy}). Beam search inactive.")

            repetition_penalty = st.slider(
                "Repetition Penalty:",
                min_value=1.0,
                max_value=3.0,
                value=1.0,
                step=0.1,
                help="Penalizes repetitive tokens. Values > 1.0 reduce degenerate looping."
            )

        with col3:
            temperature = 0.7
            top_k = 50
            top_p = 0.9
            if strategy == "sampling":
                temperature = st.slider("Temperature:", 0.1, 2.0, 0.7, 0.05, help="Controls randomness.")
                top_p = st.slider("Top-p (Nucleus):", 0.1, 1.0, 0.9, 0.05, help="Cumulative probability cutoff.")
                top_k = st.slider("Top-k:", 0, 100, 50, 5, help="Number of highest probability tokens kept.")
            else:
                st.caption("Sampling parameters (Temperature, Top-p, Top-k) are safely disabled in deterministic generation.")

    # Execution Button
    if st.button("🚀 Run CPU Inference", type="primary", width="stretch"):
        if not user_prompt.strip():
            st.error("Please enter a non-empty input prompt.")
            return

        with st.spinner("Generating autoregressive tokens on CPU ..."):
            try:
                gen_kwargs = {
                    "prompt": user_prompt,
                    "strategy": strategy,
                    "max_new_tokens": max_new_tokens,
                    "num_beams": num_beams,
                    "repetition_penalty": repetition_penalty
                }
                if strategy == "sampling":
                    gen_kwargs["temperature"] = temperature
                    gen_kwargs["top_k"] = top_k
                    gen_kwargs["top_p"] = top_p

                result = engine.generate(**gen_kwargs)

                # Display Results
                st.success("Generation completed successfully!")

                st.markdown("#### Model Generated Output:")
                st.markdown(f"""
                <div style="background: #ffffff; padding: 1.1rem; border: 1px solid #cbd5e1; border-radius: 8px; border-left: 4px solid #2563eb; font-size: 1.05rem; line-height: 1.5; color: #1e293b;">
                    {result.generated_text if result.generated_text else "<em>[Empty generation or EOS emitted immediately]</em>"}
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### Live Inference Telemetry:")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Wall-Clock Latency", f"{result.latency_ms:.1f} ms")
                m2.metric("Output Tokens", f"{result.output_tokens} tok")
                m3.metric("Throughput", f"{result.tokens_per_second:.1f} tok/s")
                m4.metric("Input Tokens", f"{result.input_tokens} tok")
                m5.metric("RAM RSS", f"{result.memory_rss_mb:.1f} MB")

                with st.expander("🔍 View Raw Generation Metadata"):
                    st.json({
                        "prompt": result.prompt,
                        "generated_text": result.generated_text,
                        "decoding_strategy": result.decoding_strategy,
                        "latency_ms": result.latency_ms,
                        "tokens_per_second": result.tokens_per_second,
                        "parameters": result.parameters,
                        "total_tokens": result.total_tokens
                    })

            except Exception as e:
                st.error(f"Inference Error: {e}")
