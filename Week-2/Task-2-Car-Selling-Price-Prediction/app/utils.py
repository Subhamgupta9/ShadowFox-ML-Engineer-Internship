"""
Utility helpers, formatting functions, and custom UI styling for CARSPHERE AI.
ShadowFox ML Engineer Internship — Task 2: Car Selling Price Prediction
"""

import streamlit as st

def apply_custom_styles():
    """Injects custom CSS for a premium, clean automotive light theme."""
    custom_css = """
    <style>
    /* Global Container Adjustments */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    
    /* Automotive Brand Header */
    .brand-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border-radius: 12px;
        padding: 24px 32px;
        color: #FFFFFF;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
        border: 1px solid #334155;
    }
    
    .brand-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .brand-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    /* Result Cards */
    .result-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px;
        margin-top: 16px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    .price-badge {
        font-size: 2.4rem;
        font-weight: 800;
        color: #0284C7;
        margin: 8px 0;
    }
    
    .price-inr-badge {
        font-size: 1.25rem;
        font-weight: 600;
        color: #0F172A;
    }
    
    .depreciation-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        background-color: #FEF3C7;
        color: #92400E;
        margin-top: 6px;
    }
    
    .kpi-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    
    .kpi-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F172A;
    }
    
    .kpi-label {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Notice Banners */
    .info-banner {
        background-color: #F0F9FF;
        border-left: 4px solid #0284C7;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        color: #0369A1;
        font-size: 0.9rem;
        margin-bottom: 20px;
    }
    
    /* Hide Default Streamlit Menu & Footer for Professional Feel */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def render_brand_header():
    """Renders the top branding navigation banner."""
    st.markdown(
        """
        <div class="brand-header">
            <h1 class="brand-title">
                🚗 CARSPHERE AI
            </h1>
            <p class="brand-subtitle">
                Intelligent Used Car Valuation & Residual Price Analytics System
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_kpi_card(title: str, value: str, subtext: str = None):
    """Renders a structured KPI card."""
    sub_html = f"<div style='font-size: 0.78rem; color: #94A3B8; margin-top: 4px;'>{subtext}</div>" if subtext else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{title}</div>
            <div class="kpi-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True
    )
