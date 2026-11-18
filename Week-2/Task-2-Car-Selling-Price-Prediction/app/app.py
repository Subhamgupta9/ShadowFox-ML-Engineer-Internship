"""
CARSPHERE AI — Intelligent Used Car Valuation & Analytics Dashboard.
ShadowFox ML Engineer Internship — Intermediate Task 2
Author: Subham Gupta (@Subhamgupta9)
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image

# Add project root and app dir to path for robust imports
ROOT_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.predict import CarPricePredictor, format_inr, VALID_FUEL_TYPES, VALID_SELLER_TYPES, VALID_TRANSMISSIONS, VALID_OWNERS, REF_YEAR

try:
    from app.utils import apply_custom_styles, render_brand_header, render_kpi_card
except ImportError:
    from utils import apply_custom_styles, render_brand_header, render_kpi_card

# Page configuration
st.set_page_config(
    page_title="CARSPHERE AI — Used Car Valuation",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply custom automotive styling
apply_custom_styles()

# Cached Model Loading
@st.cache_resource
def load_prediction_engine():
    model_path = ROOT_DIR / "models" / "car_price_model.joblib"
    return CarPricePredictor(model_path)

def main():
    render_brand_header()
    
    try:
        predictor = load_prediction_engine()
    except Exception as e:
        st.error(f"⚠️ Model artifact failed to load: {e}")
        st.stop()
        
    # Navigation Tabs
    tab_val, tab_perf, tab_eda, tab_about = st.tabs([
        "🏎️ Vehicle Valuation", 
        "📊 Model Intelligence & Benchmark", 
        "📈 Market Insights & Trends", 
        "ℹ️ About & Governance"
    ])
    
    # =========================================================================
    # TAB 1: VEHICLE VALUATION ENGINE
    # =========================================================================
    with tab_val:
        st.markdown(
            """
            <div class="info-banner">
                💡 <b>Valuation Engine:</b> Enter the vehicle's original ex-showroom price and operational specifications to estimate fair secondary market resale value calibrated against the official ShadowFox automotive benchmark.
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Initialize default session state values if not present
        if "val_price" not in st.session_state:
            st.session_state.val_price = 8.50
            st.session_state.val_year = 2015
            st.session_state.val_kms = 35000
            st.session_state.val_fuel = "Petrol"
            st.session_state.val_trans = "Manual"
            st.session_state.val_seller = "Dealer"
            st.session_state.val_owner = 0

        # Quick-Fill Vehicle Presets
        st.markdown("##### ⚡ Quick-Fill Vehicle Presets")
        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        
        with pcol1:
            if st.button("🚗 Budget Hatchback", use_container_width=True, help="Hyundai i20 / Swift (₹5.59L, 2014, 27k km, Petrol, Manual)"):
                st.session_state.val_price = 5.59
                st.session_state.val_year = 2014
                st.session_state.val_kms = 27000
                st.session_state.val_fuel = "Petrol"
                st.session_state.val_trans = "Manual"
                st.session_state.val_seller = "Dealer"
                st.session_state.val_owner = 0
                st.rerun()
                
        with pcol2:
            if st.button("🚘 Family Sedan", use_container_width=True, help="Honda City (₹9.54L, 2015, 43k km, Diesel, Manual)"):
                st.session_state.val_price = 9.54
                st.session_state.val_year = 2015
                st.session_state.val_kms = 43000
                st.session_state.val_fuel = "Diesel"
                st.session_state.val_trans = "Manual"
                st.session_state.val_seller = "Dealer"
                st.session_state.val_owner = 0
                st.rerun()

        with pcol3:
            if st.button("🚙 Premium Diesel SUV", use_container_width=True, help="Toyota Fortuner (₹30.61L, 2015, 40k km, Diesel, Auto)"):
                st.session_state.val_price = 30.61
                st.session_state.val_year = 2015
                st.session_state.val_kms = 40000
                st.session_state.val_fuel = "Diesel"
                st.session_state.val_trans = "Automatic"
                st.session_state.val_seller = "Dealer"
                st.session_state.val_owner = 0
                st.rerun()

        with pcol4:
            if st.button("🔄 Reset Defaults", use_container_width=True, help="Reset form inputs to standard defaults"):
                st.session_state.val_price = 8.50
                st.session_state.val_year = 2015
                st.session_state.val_kms = 35000
                st.session_state.val_fuel = "Petrol"
                st.session_state.val_trans = "Manual"
                st.session_state.val_seller = "Dealer"
                st.session_state.val_owner = 0
                st.rerun()

        st.markdown("<br/>", unsafe_allow_html=True)
        
        with st.form("valuation_form"):
            col1, col2 = st.columns(2, gap="large")
            
            with col1:
                st.subheader("📋 Core Vehicle Specifications")
                
                present_price = st.number_input(
                    "Original Ex-Showroom Price (in Lakhs ₹)",
                    min_value=0.50,
                    max_value=100.00,
                    value=float(st.session_state.val_price),
                    step=0.25,
                    help="The original invoice / showroom price when the vehicle was brand new (e.g., 8.50 = ₹8,50,000)."
                )
                
                year = st.slider(
                    "Manufacturing Year",
                    min_value=2003,
                    max_value=2018,
                    value=int(st.session_state.val_year),
                    step=1,
                    help=f"Vehicle model manufacturing year (2003–2018). Calibrated against reference transaction horizon {REF_YEAR} to guarantee non-negative vehicle age."
                )
                
                kms_driven = st.number_input(
                    "Total Odometer Reading (Kilometers Driven)",
                    min_value=500,
                    max_value=500000,
                    value=int(st.session_state.val_kms),
                    step=2500,
                    help="Total distance accumulated by the vehicle in kilometers."
                )
                
            with col2:
                st.subheader("⚙️ Technical & Ownership Details")
                
                fuel_idx = VALID_FUEL_TYPES.index(st.session_state.val_fuel) if st.session_state.val_fuel in VALID_FUEL_TYPES else 0
                fuel_type = st.selectbox(
                    "Fuel Type",
                    options=VALID_FUEL_TYPES,
                    index=fuel_idx,
                    help="Primary fuel category (Petrol, Diesel, or CNG)."
                )
                
                trans_idx = VALID_TRANSMISSIONS.index(st.session_state.val_trans) if st.session_state.val_trans in VALID_TRANSMISSIONS else 0
                transmission = st.selectbox(
                    "Transmission Type",
                    options=VALID_TRANSMISSIONS,
                    index=trans_idx,
                    help="Transmission mechanism (Manual or Automatic)."
                )
                
                seller_idx = VALID_SELLER_TYPES.index(st.session_state.val_seller) if st.session_state.val_seller in VALID_SELLER_TYPES else 0
                seller_type = st.selectbox(
                    "Seller Channel",
                    options=VALID_SELLER_TYPES,
                    index=seller_idx,
                    help="Whether the vehicle is sold through a verified Dealership or directly by an Individual."
                )
                
                owner_labels = {0: "0 (First Owner / Single Handed)", 1: "1 (Second Owner)", 2: "2 (Third Owner)", 3: "3 (Fourth+ Owner)"}
                owner_idx = VALID_OWNERS.index(st.session_state.val_owner) if st.session_state.val_owner in VALID_OWNERS else 0
                owner_val = st.selectbox(
                    "Number of Previous Owners",
                    options=VALID_OWNERS,
                    format_func=lambda x: owner_labels.get(x, str(x)),
                    index=owner_idx,
                    help="Count of registered prior owners on the vehicle RC."
                )
                
            submit_btn = st.form_submit_button("🔍 Calculate Estimated Resale Price", type="primary", use_container_width=True)
            
        if submit_btn:
            input_dict = {
                'Present_Price': present_price,
                'Kms_Driven': kms_driven,
                'Fuel_Type': fuel_type,
                'Seller_Type': seller_type,
                'Transmission': transmission,
                'Owner': owner_val,
                'Year': year
            }
            
            try:
                res = predictor.predict(input_dict)
                pred_lakhs = res['predicted_price_lakhs']
                pred_inr = res['predicted_price_inr']
                deprec_pct = res['depreciation_percentage']
                veh_age = res['input_features']['Vehicle_Age']
                
                st.markdown("<hr style='margin: 24px 0 16px 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)
                
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.5px; color: #64748B; font-weight: 600;">
                            Estimated Fair Resale Valuation
                        </div>
                        <div class="price-badge">
                            ₹{pred_lakhs:.2f} Lakhs
                        </div>
                        <div class="price-inr-badge">
                            {pred_inr}
                        </div>
                        <div class="depreciation-pill">
                            📉 Projected Depreciation: {deprec_pct:.1f}% from showroom price
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Breakdown KPIs
                kpi1, kpi2, kpi3 = st.columns(3)
                with kpi1:
                    render_kpi_card("Original Showroom Price", f"₹{present_price:.2f} L", f"{format_inr(present_price)}")
                with kpi2:
                    render_kpi_card("Calculated Vehicle Age", f"{int(veh_age)} Years", f"Model Year {year} (Baseline {REF_YEAR})")
                with kpi3:
                    total_deprec = max(0.0, present_price - pred_lakhs)
                    render_kpi_card("Total Value Depreciated", f"₹{total_deprec:.2f} L", f"{format_inr(total_deprec)}")
                    
            except Exception as ex:
                st.error(f"Valuation computation error: {ex}")

    # =========================================================================
    # TAB 2: MODEL PERFORMANCE & BENCHMARK
    # =========================================================================
    with tab_perf:
        st.subheader("🏆 Model Evaluation & Algorithm Benchmarks")
        st.write(
            "Evaluation of five regression architectures using 5-Fold Cross-Validation on the training partition "
            "and final assessment on the held-out 20% test partition."
        )
        
        # Performance KPIs
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_kpi_card("Unseen Test R²", "0.812", "81.2% Variance Explained")
        with m2:
            render_kpi_card("Test MAE", "₹1.00 L", "Mean Absolute Error: ₹1,00,373")
        with m3:
            render_kpi_card("Test RMSE", "₹2.20 L", "Root Mean Squared Error")
        with m4:
            render_kpi_card("5-Fold CV R²", "0.897 ± 0.092", "Cross-Validation Stability")
            
        st.markdown("<br/>", unsafe_allow_html=True)
        
        # Benchmark CSV Table
        csv_path = ROOT_DIR / "reports" / "experiment_results.csv"
        if csv_path.exists():
            df_metrics = pd.read_csv(csv_path)
            st.markdown("#### 📋 Detailed Benchmark Comparison Table")
            st.dataframe(
                df_metrics.style.format({
                    "CV_R2_Mean": "{:.4f}",
                    "CV_R2_Std": "{:.4f}",
                    "CV_MAE_Mean": "₹{:.3f}L",
                    "CV_RMSE_Mean": "₹{:.3f}L",
                    "Test_R2": "{:.4f}",
                    "Test_MAE": "₹{:.3f}L",
                    "Test_MSE": "{:.4f}",
                    "Test_RMSE": "₹{:.3f}L",
                    "Train_R2": "{:.4f}"
                }).highlight_max(subset=['Test_R2', 'CV_R2_Mean'], color='#D1FAE5')
                  .highlight_min(subset=['Test_MAE', 'Test_RMSE'], color='#FEF3C7'),
                use_container_width=True
            )
            
        # Benchmark Visualizations
        col_c1, col_c2 = st.columns(2)
        fig_comp = ROOT_DIR / "reports" / "figures" / "model_comparison_chart.png"
        fig_res = ROOT_DIR / "reports" / "figures" / "residuals_analysis.png"
        
        if fig_comp.exists():
            with col_c1:
                st.image(str(fig_comp), caption="Model Benchmark Comparison (CV vs Test R²)", use_container_width=True)
        if fig_res.exists():
            with col_c2:
                st.image(str(fig_res), caption="Actual vs Predicted & Residual Error Analysis", use_container_width=True)
                
        fig_fi = ROOT_DIR / "reports" / "figures" / "feature_importance.png"
        if fig_fi.exists():
            st.image(str(fig_fi), caption="Final Feature Importance Attribution (Extra Trees Regressor)", use_container_width=True)

    # =========================================================================
    # TAB 3: MARKET INSIGHTS & EXPLORATORY DATA ANALYSIS
    # =========================================================================
    with tab_eda:
        st.subheader("📈 Automotive Secondary Market Insights")
        st.write("Exploratory Data Analysis generated from the official ShadowFox pre-owned vehicle dataset.")
        
        c_e1, c_e2 = st.columns(2)
        f_price = ROOT_DIR / "reports" / "figures" / "price_distribution.png"
        f_show = ROOT_DIR / "reports" / "figures" / "showroom_vs_selling.png"
        
        if f_price.exists():
            with c_e1:
                st.image(str(f_price), caption="Distribution of Resale Prices & Boxplot Outliers", use_container_width=True)
        if f_show.exists():
            with c_e2:
                st.image(str(f_show), caption="Showroom Price vs Resale Price Linear Trend", use_container_width=True)
                
        c_e3, c_e4 = st.columns(2)
        f_age = ROOT_DIR / "reports" / "figures" / "age_vs_selling.png"
        f_fuel = ROOT_DIR / "reports" / "figures" / "fuel_transmission_distribution.png"
        
        if f_age.exists():
            with c_e3:
                st.image(str(f_age), caption="Depreciation Curve: Resale Value Across Years of Service", use_container_width=True)
        if f_fuel.exists():
            with c_e4:
                st.image(str(f_fuel), caption="Average Selling Price by Fuel and Transmission Types", use_container_width=True)

    # =========================================================================
    # TAB 4: ABOUT & GOVERNANCE
    # =========================================================================
    with tab_about:
        st.subheader("ℹ️ Project Portfolio & Machine Learning Governance")
        
        st.markdown(
            """
            ### ShadowFox Machine Learning Engineer Internship
            - **Intern:** [Subham Gupta](https://github.com/Subhamgupta9)
            - **Track:** Intermediate Task 2 — Car Selling Price Prediction & Analysis
            - **Organization:** [ShadowFox](https://shadowfox.in/)
            - **Repository:** [`ShadowFox-ML-Engineer-Internship`](https://github.com/Subhamgupta9/ShadowFox-ML-Engineer-Internship)
            
            ---
            
            ### 📜 Dataset Lineage & Historical Reference Horizon
            The model is trained on the official ShadowFox used vehicle dataset consisting of **299 verified unique records** (2 exact duplicate listings removed to eliminate train-test data leakage).
            
            > **Historical Reference Year Policy:** The latest vehicle manufacturing year in the secondary market dataset is **2018**. Therefore, all vehicle age calculations are modeled as:
            > $$\text{Vehicle\_Age} = 2018 - \text{Year}$$
            > Valid manufacturing years are strictly constrained between **2003 and 2018** (resulting in non-negative vehicle ages from 0 to 15 years). Years past 2018 (2019–2026) are rejected to prevent negative vehicle ages and out-of-distribution extrapolation.
            
            ### 🛡️ System Boundaries & Limitations
            1. **Consistent Historical Horizon:** Valuations reflect secondary market conditions of the historical capture era (up to 2018). Post-2020 macroeconomic inflation, supply-chain adjustments, and electric vehicle (EV) market shifts are outside the historical training distribution.
            2. **Segment Concentration:** High precision is maintained across mainstream hatchbacks, sedans, and compact SUVs (₹1L to ₹15L). Extreme exotic vehicles ($> \text{₹}50\text{L}$) carry wider predictive variance due to low representation.
            3. **Salvage Valuation Floor:** An economic salvage floor of ₹10,000 (0.10 Lakhs) is enforced to prevent unphysical negative values on heavily aged or high-mileage assets.
            """
        )

if __name__ == "__main__":
    main()
