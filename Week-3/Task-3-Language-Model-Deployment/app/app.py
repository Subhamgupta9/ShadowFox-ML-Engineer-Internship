"""
NEXUS NLP — Language Model Deployment & Intelligence Evaluation Lab
Streamlit Web Application Entrypoint

Author: Subham Gupta (ShadowFox ML Engineer Intern)
Week 3: Task 3 - Language Model Deployment & Intelligence Evaluation Lab
Model: google/flan-t5-small (~76.96M Parameters)
Hardware: CPU Only (Intel x86_64)
"""

import os
import sys
import streamlit as st

# Configure page layout and metadata
st.set_page_config(
    page_title="NEXUS NLP — Language Model Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_engine import NexusModelEngine
try:
    from components.ui_header import render_header
    from components.tab_lab import render_tab_lab
    from components.tab_benchmark import render_tab_benchmark
    from components.tab_comparator import render_tab_comparator
    from components.tab_inspector import render_tab_inspector
except ImportError:
    from app.components.ui_header import render_header
    from app.components.tab_lab import render_tab_lab
    from app.components.tab_benchmark import render_tab_benchmark
    from app.components.tab_comparator import render_tab_comparator
    from app.components.tab_inspector import render_tab_inspector


# ==============================================================================
# Model Engine Caching Singleton
# ==============================================================================

@st.cache_resource(show_spinner="Loading FLAN-T5-small model into memory (CPU) ...")
def get_model_engine():
    """
    Initializes and caches the NexusModelEngine singleton instance.
    Guarantees zero redundant re-downloads and minimal RAM footprint.
    """
    cache_dir = os.path.join(PROJECT_ROOT, "model_cache")
    engine = NexusModelEngine(cache_dir=cache_dir, device="cpu")
    engine.load()
    return engine


# ==============================================================================
# Custom CSS for Professional Research Lab Theme
# ==============================================================================

def inject_custom_styles():
    st.markdown("""
    <style>
        /* General Font and Spacing */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            max-width: 1320px;
        }

        /* Metric Cards */
        div[data-testid="stMetric"] {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 0.75rem 1rem;
            border-radius: 8px;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.82rem;
            color: #64748b;
            font-weight: 600;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.4rem;
            color: #0f172a;
            font-weight: 700;
        }

        /* Tabs Styling */
        button[data-baseweb="tab"] {
            font-size: 0.95rem;
            font-weight: 600;
            padding: 0.6rem 1.2rem;
        }

        /* Text Area */
        textarea {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
            font-size: 0.92rem !important;
        }

        /* Footer */
        .footer-text {
            text-align: center;
            font-size: 0.82rem;
            color: #64748b;
            margin-top: 2.5rem;
            padding-top: 1rem;
            border-top: 1px solid #e2e8f0;
        }
    </style>
    """, unsafe_allow_html=True)


def main():
    inject_custom_styles()

    # Initialize / Load Engine
    try:
        engine = get_model_engine()
        specs = engine.get_specs()
    except Exception as e:
        st.error(f"Failed to initialize language model engine: {e}")
        return

    # Render Header
    render_header(specs)

    # Reports Directory Path
    reports_dir = os.path.join(PROJECT_ROOT, "reports")

    # Major Application Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 Interactive NLP Lab",
        "📊 Multi-Task Benchmark Dashboard",
        "⚔️ Decoding Strategy Comparator",
        "🛡️ Architecture & Safety Inspector"
    ])

    with tab1:
        render_tab_lab(engine)

    with tab2:
        render_tab_benchmark(reports_dir)

    with tab3:
        render_tab_comparator(engine)

    with tab4:
        render_tab_inspector(engine)

    # Footer
    st.markdown("""
    <div class="footer-text">
        <strong>NEXUS NLP: Language Model Deployment & Intelligence Evaluation Lab</strong> &bull;
        Week 3 Advanced Task 3 &bull; ShadowFox Machine Learning Engineer Internship<br>
        Built by <strong>Subham Gupta</strong> (<a href="https://github.com/Subhamgupta9" target="_blank" style="color: #2563eb; text-decoration: none;">@Subhamgupta9</a>) &bull;
        Model: <code>google/flan-t5-small</code> (76.96M Params) &bull; Executed on CPU Only
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
