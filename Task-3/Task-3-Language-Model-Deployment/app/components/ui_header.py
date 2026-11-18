"""
NEXUS NLP — Streamlit Header & Status Component
Renders the research laboratory header, hardware telemetry badges, and advisory notices.

Author: Subham Gupta (ShadowFox ML Engineer Intern)
"""

import streamlit as st


def render_header(specs: dict):
    """Renders the top laboratory hero section, badges, and safety disclaimer."""

    st.markdown("""
    <div style="padding: 1.2rem 1.4rem; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; margin-bottom: 1.2rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
            <div>
                <h1 style="margin: 0; font-size: 1.85rem; font-weight: 700; color: #0f172a; letter-spacing: -0.02em;">
                    🔬 NEXUS NLP: Language Model Evaluation & Intelligence Lab
                </h1>
                <p style="margin: 0.35rem 0 0 0; font-size: 0.95rem; color: #475569;">
                    Week 3 Advanced Task 3 | ShadowFox Machine Learning Engineer Internship &bull;
                    Author: <strong style="color: #1e293b;">Subham Gupta</strong>
                    (<a href="https://github.com/Subhamgupta9" target="_blank" style="color: #2563eb; text-decoration: none;">@Subhamgupta9</a>)
                </p>
            </div>
            <div style="display: flex; gap: 0.4rem; flex-wrap: wrap;">
                <span style="background: #e0f2fe; color: #0369a1; padding: 0.25rem 0.65rem; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">
                    Model: google/flan-t5-small
                </span>
                <span style="background: #f1f5f9; color: #334155; padding: 0.25rem 0.65rem; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">
                    Params: 76.96M
                </span>
                <span style="background: #ecfdf5; color: #047857; padding: 0.25rem 0.65rem; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">
                    Device: CPU Only
                </span>
                <span style="background: #fef3c7; color: #b45309; padding: 0.25rem 0.65rem; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">
                    Activation: gated-gelu
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Permanent Research Safety & Transparency Banner
    st.info(
        "**Research Notice & Output Reliability:** `google/flan-t5-small` is an ultra-compact (~77M parameter) "
        "instruction-tuned model. Generated outputs are generated on local CPU and may exhibit factual hallucination, "
        "arithmetic errors, or over-simplification. Do not rely on generations for mission-critical decisions without human oversight."
    )
