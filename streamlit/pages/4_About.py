"""
About Page - Comprehensive Project Documentation

Provides information for both business stakeholders and data science practitioners
following the Snowflake Streamlit About Section Guide framework.
"""
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.styling import (
    inject_custom_css,
    render_story_section,
    render_learn_more_panel,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_ACCENT,
    COLOR_SUCCESS,
    COLOR_WARNING,
    BG_CARD,
    BG_HOVER,
    FONT_MONO
)

# --- Page Config ---
st.set_page_config(
    page_title="About | MMM ROI Engine",
    layout="wide"
)

inject_custom_css()


def render_data_source_card(table_name: str, description: str, badge_type: str, key_fields: list = None):
    """Render a styled data source card with badge."""
    badge_colors = {
        "ERP": COLOR_SUCCESS,
        "CRM": COLOR_PRIMARY,
        "AD_PLATFORM": COLOR_ACCENT,
        "EXTERNAL": COLOR_WARNING,
        "MODEL": "#9B59B6"  # Purple for model outputs
    }
    badge_color = badge_colors.get(badge_type, COLOR_PRIMARY)
    
    # Build key fields as simple text
    fields_text = ""
    if key_fields:
        fields_text = ', '.join(key_fields)
    
    html = f'<div style="background: {BG_CARD}; border-radius: 10px; padding: 1rem; margin-bottom: 0.75rem; border: 1px solid rgba(255,255,255,0.1);">'
    html += f'<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">'
    html += f'<code style="color: white; font-size: 0.85rem;">{table_name}</code>'
    html += f'<span style="background: {badge_color}22; color: {badge_color}; font-size: 0.65rem; padding: 0.15rem 0.5rem; border-radius: 12px; font-weight: 600; border: 1px solid {badge_color}44;">{badge_type}</span>'
    html += '</div>'
    html += f'<div style="color: rgba(255,255,255,0.7); font-size: 0.85rem; line-height: 1.4;">{description}</div>'
    
    if fields_text:
        html += f'<div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid rgba(255,255,255,0.1);">'
        html += f'<div style="color: rgba(255,255,255,0.5); font-size: 0.7rem; text-transform: uppercase; margin-bottom: 0.25rem;">Key Fields</div>'
        html += f'<code style="font-size: 0.75rem; color: rgba(255,255,255,0.6);">{fields_text}</code>'
        html += '</div>'
    
    html += '</div>'
    return html


def main():
    # =========================================================================
    # 1. HEADER
    # =========================================================================
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>Global B2B Marketing Mix & ROI Engine</h1>
        <p style="color: rgba(255,255,255,0.6); font-size: 1.2rem;">
            Powered by Snowflake Snowpark ML & Cortex AI
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # =========================================================================
    # 2. OVERVIEW (Problem + Solution - Two Columns)
    # =========================================================================
    st.markdown("## Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### The Problem")
        st.markdown("""
        For global B2B enterprises, marketing data is **fragmented across multiple systems**:
        
        - **Ad Platforms** (Sprinklr, Google, LinkedIn) → Impressions, Clicks, Spend
        - **CRM** (Salesforce) → Pipeline, Opportunities, Win Rates
        - **ERP** (SAP) → Booked Revenue, Invoice Data
        
        The result? **No clear line of sight** from marketing investment to actual revenue.
        
        Traditional attribution fails because:
        - B2B sales cycles span **6-18 months**
        - Revenue flows through **distributor partners**
        - Marketing impact is **lagged and indirect**
        
        This leads to inefficient **"peanut butter" budget spreading** instead of 
        data-driven allocation that maximizes ROI.
        """)
    
    with col2:
        st.markdown("### The Solution")
        st.markdown(f"""
        <div style="background: {BG_CARD}; border-radius: 12px; padding: 1.5rem; 
                    border-left: 4px solid {COLOR_PRIMARY};">
            <p style="color: rgba(255,255,255,0.85); line-height: 1.6;">
                A <strong>Marketing Mix Model (MMM)</strong> that:
            </p>
            <ul style="color: rgba(255,255,255,0.8); line-height: 1.8; padding-left: 1.2rem;">
                <li>Unifies data from all sources</li>
                <li>Calculates true ROI by channel</li>
                <li>Provides confidence intervals</li>
                <li>Enables budget simulation</li>
                <li>Answers questions in plain English</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================================================================
    # 3. DATA ARCHITECTURE (Three Columns with Badges)
    # =========================================================================
    st.markdown("## Data Architecture")
    st.markdown("*All data flows through Snowflake with full governance and lineage tracking.*")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Internal Data")
        st.markdown(render_data_source_card(
            "SPRINKLR_SPEND",
            "Daily marketing spend, impressions, and clicks from all ad platforms aggregated by Sprinklr.",
            "AD_PLATFORM",
            ["AD_DATE", "CAMPAIGN_ID", "SPEND_USD", "IMPRESSIONS"]
        ), unsafe_allow_html=True)
        
        st.markdown(render_data_source_card(
            "SFDC_OPPORTUNITIES",
            "Salesforce pipeline data tracking deals from lead to close with campaign attribution.",
            "CRM",
            ["OPP_ID", "STAGE", "AMOUNT_USD", "CAMPAIGN_ID"]
        ), unsafe_allow_html=True)
        
        st.markdown(render_data_source_card(
            "SAP_REVENUE",
            "Invoiced revenue from SAP including distributor sales and direct bookings.",
            "ERP",
            ["INVOICE_NUM", "POSTING_DATE", "NET_VALUE_USD"]
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### External Data")
        st.markdown(render_data_source_card(
            "MARKET_INDICATOR",
            "Weekly macro-economic signals including PMI index and competitor share of voice.",
            "EXTERNAL",
            ["WEEK_START", "REGION", "PMI_INDEX", "COMPETITOR_SOV"]
        ), unsafe_allow_html=True)
        
        st.markdown(render_data_source_card(
            "CAMPAIGN_BRIEFS",
            "PDF campaign strategy documents indexed for RAG-based Q&A via Cortex Search.",
            "EXTERNAL",
            ["CAMPAIGN_ID", "BRIEF_PDF", "CREATIVE_STRATEGY"]
        ), unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {BG_CARD}, {BG_HOVER}); 
                    border-radius: 10px; padding: 1rem; border: 1px dashed rgba(255,255,255,0.2);">
            <div style="color: rgba(255,255,255,0.5); font-size: 0.8rem; text-align: center;">
                Data refreshed weekly via Snowflake Tasks
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("#### Model Outputs")
        st.markdown(render_data_source_card(
            "MMM.MODEL_RESULTS",
            "Channel-level ROI with 90% confidence intervals, marginal ROI, and optimal spend recommendations.",
            "MODEL",
            ["CHANNEL", "ROI", "ROI_CI_LOWER", "MARGINAL_ROI"]
        ), unsafe_allow_html=True)
        
        st.markdown(render_data_source_card(
            "MMM.RESPONSE_CURVES",
            "Spend-to-revenue curves for visualization with efficiency zone classifications.",
            "MODEL",
            ["CHANNEL", "SPEND", "PREDICTED_REVENUE", "EFFICIENCY_ZONE"]
        ), unsafe_allow_html=True)
        
        st.markdown(render_data_source_card(
            "MMM.MODEL_METADATA",
            "Model configuration, quality metrics (R², CV MAPE), and hyperparameters.",
            "MODEL",
            ["MODEL_VERSION", "R2_INSAMPLE", "MAPE_CV"]
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================================================================
    # 4. HOW IT WORKS (Scrollable Sections)
    # =========================================================================
    st.markdown("## How It Works")
    
    # -------------------------------------------------------------------------
    # EXECUTIVE OVERVIEW SECTION
    # -------------------------------------------------------------------------
    st.markdown("### Executive Overview")
    
    st.markdown("#### Why Traditional Attribution Fails for B2B")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="background: {BG_CARD}; border-radius: 12px; padding: 1.5rem; height: 100%;">
            <h4 style="color: {COLOR_ACCENT};">Last-Click Attribution</h4>
            <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">
                Credits the final touchpoint before conversion. In B2B, that's often a 
                salesperson—ignoring the 6 months of marketing that created the opportunity.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: {BG_CARD}; border-radius: 12px; padding: 1.5rem; height: 100%;">
            <h4 style="color: {COLOR_ACCENT};">MTA / Multi-Touch</h4>
            <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">
                Requires user-level tracking, which is impossible when 60% of revenue flows 
                through distributors who never touch your digital properties.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: {BG_CARD}; border-radius: 12px; padding: 1.5rem; height: 100%;">
            <h4 style="color: {COLOR_SUCCESS};">Marketing Mix Modeling</h4>
            <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">
                Uses <strong>aggregate data</strong> (total spend vs. total revenue) with 
                statistical regression. Works for long cycles and distributor sales.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("#### How MMM Connects Spend to Revenue")
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {BG_CARD}, #1e2738); 
                border-radius: 12px; padding: 2rem; margin: 1rem 0;">
        <p style="color: rgba(255,255,255,0.85); font-size: 1.05rem; line-height: 1.7;">
            Think of it like tracking the <strong>"echo"</strong> of your marketing:
        </p>
        <ol style="color: rgba(255,255,255,0.8); line-height: 2; padding-left: 1.5rem;">
            <li><strong>Carryover Effect:</strong> A LinkedIn ad today still influences buyers 
                4-6 weeks later. The model learns how quickly each channel's effect "decays."</li>
            <li><strong>Diminishing Returns:</strong> The first $50K on Google Search is efficient. 
                The 10th $50K reaches the same people again—less incremental value.</li>
            <li><strong>Control for Noise:</strong> Q4 revenue is always high (holidays). 
                The model isolates true marketing impact from seasonality and economic cycles.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### Business Value Delivered")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {COLOR_PRIMARY}15, {COLOR_SECONDARY}15); 
                    border: 1px solid {COLOR_PRIMARY}; border-radius: 12px; padding: 1.25rem; text-align: center;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;"></div>
            <div style="color: white; font-weight: 600;">Unified Attribution</div>
            <div style="color: rgba(255,255,255,0.6); font-size: 0.8rem; margin-top: 0.5rem;">
                Single ROI per Channel × Region × Product
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {COLOR_SUCCESS}15, {COLOR_SUCCESS}08); 
                    border: 1px solid {COLOR_SUCCESS}; border-radius: 12px; padding: 1.25rem; text-align: center;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;"></div>
            <div style="color: white; font-weight: 600;">Confidence Intervals</div>
            <div style="color: rgba(255,255,255,0.6); font-size: 0.8rem; margin-top: 0.5rem;">
                "LinkedIn ROI: 3.2x [2.8 - 3.6]"
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {COLOR_ACCENT}15, {COLOR_ACCENT}08); 
                    border: 1px solid {COLOR_ACCENT}; border-radius: 12px; padding: 1.25rem; text-align: center;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;"></div>
            <div style="color: white; font-weight: 600;">Marginal ROI</div>
            <div style="color: rgba(255,255,255,0.6); font-size: 0.8rem; margin-top: 0.5rem;">
                Where should the NEXT dollar go?
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {COLOR_WARNING}15, {COLOR_WARNING}08); 
                    border: 1px solid {COLOR_WARNING}; border-radius: 12px; padding: 1.25rem; text-align: center;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;"></div>
            <div style="color: white; font-weight: 600;">What-If Simulation</div>
            <div style="color: rgba(255,255,255,0.6); font-size: 0.8rem; margin-top: 0.5rem;">
                Predict lift before reallocation
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # -------------------------------------------------------------------------
    # TECHNICAL DEEP-DIVE SECTION
    # -------------------------------------------------------------------------
    st.markdown("### Technical Deep-Dive")
    
    st.markdown("#### MMM Algorithm Architecture")
    
    st.markdown("""
    The model implements techniques from **Meta's Robyn** and **Google's LightweightMMM** 
    adapted for Snowflake's compute environment.
    """)
    
    # Algorithm components table
    st.markdown(f"""
    <table style="width: 100%; border-collapse: collapse; margin: 1rem 0;">
        <tr style="background: {BG_CARD};">
            <th style="padding: 0.75rem; text-align: left; color: {COLOR_PRIMARY}; border-bottom: 1px solid rgba(255,255,255,0.1);">Component</th>
            <th style="padding: 0.75rem; text-align: left; color: {COLOR_PRIMARY}; border-bottom: 1px solid rgba(255,255,255,0.1);">Implementation</th>
            <th style="padding: 0.75rem; text-align: left; color: {COLOR_PRIMARY}; border-bottom: 1px solid rgba(255,255,255,0.1);">Purpose</th>
        </tr>
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
            <td style="padding: 0.75rem; color: white;">Geometric Adstock</td>
            <td style="padding: 0.75rem; color: rgba(255,255,255,0.7);"><code>x_eff[t] = x[t] + θ·x_eff[t-1]</code></td>
            <td style="padding: 0.75rem; color: rgba(255,255,255,0.7);">Models carryover effect (θ = decay rate)</td>
        </tr>
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
            <td style="padding: 0.75rem; color: white;">Hill Saturation</td>
            <td style="padding: 0.75rem; color: rgba(255,255,255,0.7);"><code>x^α / (x^α + γ^α)</code></td>
            <td style="padding: 0.75rem; color: rgba(255,255,255,0.7);">Captures diminishing returns (γ = half-saturation)</td>
        </tr>
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
            <td style="padding: 0.75rem; color: white;">Nevergrad Optimization</td>
            <td style="padding: 0.75rem; color: rgba(255,255,255,0.7);">TwoPointsDE (500 iterations)</td>
            <td style="padding: 0.75rem; color: rgba(255,255,255,0.7);">Finds optimal θ, α, γ per channel</td>
        </tr>
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
            <td style="padding: 0.75rem; color: white;">Ridge Regression</td>
            <td style="padding: 0.75rem; color: rgba(255,255,255,0.7);">L2 penalty (α=1.0), positive constraints</td>
            <td style="padding: 0.75rem; color: rgba(255,255,255,0.7);">Estimates coefficients with regularization</td>
        </tr>
        <tr>
            <td style="padding: 0.75rem; color: white;">Bootstrap CI</td>
            <td style="padding: 0.75rem; color: rgba(255,255,255,0.7);">100 iterations, 90% percentile</td>
            <td style="padding: 0.75rem; color: rgba(255,255,255,0.7);">Uncertainty quantification on ROI</td>
        </tr>
    </table>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Training Workflow")
        st.markdown(f"""
        <div style="background: {BG_CARD}; border-radius: 12px; padding: 1.5rem;">
            <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <span style="background: {COLOR_PRIMARY}; color: white; width: 24px; height: 24px; 
                                 border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                                 font-size: 0.75rem; font-weight: 600;">1</span>
                    <span style="color: rgba(255,255,255,0.85);">Load from <code>V_MMM_INPUT_WEEKLY</code></span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <span style="background: {COLOR_PRIMARY}; color: white; width: 24px; height: 24px; 
                                 border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                                 font-size: 0.75rem; font-weight: 600;">2</span>
                    <span style="color: rgba(255,255,255,0.85);">Add Fourier seasonality + trend controls</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <span style="background: {COLOR_PRIMARY}; color: white; width: 24px; height: 24px; 
                                 border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                                 font-size: 0.75rem; font-weight: 600;">3</span>
                    <span style="color: rgba(255,255,255,0.85);">Optimize adstock/saturation params (Nevergrad)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <span style="background: {COLOR_PRIMARY}; color: white; width: 24px; height: 24px; 
                                 border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                                 font-size: 0.75rem; font-weight: 600;">4</span>
                    <span style="color: rgba(255,255,255,0.85);">Fit Ridge regression with transformed features</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <span style="background: {COLOR_PRIMARY}; color: white; width: 24px; height: 24px; 
                                 border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                                 font-size: 0.75rem; font-weight: 600;">5</span>
                    <span style="color: rgba(255,255,255,0.85);">Bootstrap for 90% confidence intervals</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <span style="background: {COLOR_SUCCESS}; color: white; width: 24px; height: 24px; 
                                 border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                                 font-size: 0.75rem; font-weight: 600;">OK</span>
                    <span style="color: rgba(255,255,255,0.85);">Write to <code>MMM.MODEL_RESULTS</code></span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Model Quality Metrics")
        st.markdown(f"""
        <div style="background: {BG_CARD}; border-radius: 12px; padding: 1.5rem;">
            <table style="width: 100%;">
                <tr>
                    <td style="padding: 0.5rem 0; color: rgba(255,255,255,0.7);">In-Sample R²</td>
                    <td style="padding: 0.5rem 0; color: white; text-align: right; font-weight: 600;">Target: > 0.85</td>
                </tr>
                <tr>
                    <td style="padding: 0.5rem 0; color: rgba(255,255,255,0.7);">CV MAPE</td>
                    <td style="padding: 0.5rem 0; color: white; text-align: right; font-weight: 600;">Target: &lt; 15%</td>
                </tr>
                <tr>
                    <td style="padding: 0.5rem 0; color: rgba(255,255,255,0.7);">Significant Channels</td>
                    <td style="padding: 0.5rem 0; color: white; text-align: right; font-weight: 600;">Target: > 60%</td>
                </tr>
                <tr>
                    <td style="padding: 0.5rem 0; color: rgba(255,255,255,0.7);">Training Time</td>
                    <td style="padding: 0.5rem 0; color: white; text-align: right; font-weight: 600;">~5-10 minutes</td>
                </tr>
            </table>
            <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1);">
                <div style="color: rgba(255,255,255,0.5); font-size: 0.8rem;">
                    <strong>Validation:</strong> Time-series CV with 52-week train / 13-week test windows. 
                    Never peeks at future data.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### Hyperparameter Configuration")
    
    st.code("""
config = MMMConfig(
    geo_level="SUPER_REGION",      # Granularity: SUPER_REGION | REGION | COUNTRY
    product_level="SEGMENT",       # Granularity: SEGMENT | DIVISION | CATEGORY
    nevergrad_budget=500,          # Evolutionary optimization iterations
    n_bootstrap=100,               # Bootstrap samples for CI
    confidence_level=0.90,         # 90% confidence intervals
    cv_train_weeks=52,             # 1-year rolling training window
    cv_test_weeks=13,              # 1-quarter holdout for validation
    budget_change_limit=0.30       # ±30% reallocation constraint
)
    """, language="python")
    
    st.markdown(f"""
    <div style="background: {BG_CARD}; border-radius: 8px; padding: 1rem; margin-top: 1rem;">
        <div style="color: {COLOR_PRIMARY}; font-weight: 600; margin-bottom: 0.5rem;">Notebook Location</div>
        <code style="color: rgba(255,255,255,0.7);">notebooks/01_mmm_training.ipynb</code>
        <div style="color: rgba(255,255,255,0.5); font-size: 0.8rem; margin-top: 0.5rem;">
            Runs on Snowflake Notebooks or SPCS with Python 3.11
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================================================================
    # 5. CORTEX AI CAPABILITIES
    # =========================================================================
    st.markdown("## Cortex AI Capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Cortex Analyst (Structured Data)")
        st.markdown("""
        Ask questions in plain English about your marketing data. 
        The semantic model translates to SQL automatically.
        """)
        
        st.markdown(f"""
        <div style="background: {BG_CARD}; border-radius: 12px; padding: 1.5rem; margin: 1rem 0;">
            <div style="color: {COLOR_PRIMARY}; font-weight: 600; margin-bottom: 1rem;">Semantic Model Scope</div>
            <div style="margin-bottom: 1rem;">
                <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem; text-transform: uppercase;">Tables</div>
                <code style="color: rgba(255,255,255,0.8);">V_MMM_INPUT_WEEKLY</code>, 
                <code style="color: rgba(255,255,255,0.8);">V_ROI_BY_CHANNEL</code>
            </div>
            <div style="margin-bottom: 1rem;">
                <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem; text-transform: uppercase;">Measures</div>
                <span style="color: rgba(255,255,255,0.8);">Spend, Revenue, ROAS, Impressions, Clicks, PMI, SOV</span>
            </div>
            <div>
                <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem; text-transform: uppercase;">Dimensions</div>
                <span style="color: rgba(255,255,255,0.8);">Channel, Super-Region, Region, Country, Week</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Sample Questions:**")
        st.markdown(f"""
        <div style="background: {BG_HOVER}; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
            <div style="color: rgba(255,255,255,0.9); font-style: italic;">
                "Show me ROAS by business segment for APAC"
            </div>
        </div>
        <div style="background: {BG_HOVER}; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
            <div style="color: rgba(255,255,255,0.9); font-style: italic;">
                "Compare LinkedIn vs Google ROAS for Germany"
            </div>
        </div>
        <div style="background: {BG_HOVER}; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
            <div style="color: rgba(255,255,255,0.9); font-style: italic;">
                "What's the marginal ROI by channel for Western Europe?"
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Cortex Search (Unstructured Data)")
        st.markdown("""
        Search campaign briefs and strategy documents using natural language. 
        Powered by RAG (Retrieval-Augmented Generation).
        """)
        
        st.markdown(f"""
        <div style="background: {BG_CARD}; border-radius: 12px; padding: 1.5rem; margin: 1rem 0;">
            <div style="color: {COLOR_PRIMARY}; font-weight: 600; margin-bottom: 1rem;">Document Index</div>
            <div style="margin-bottom: 1rem;">
                <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem; text-transform: uppercase;">Service</div>
                <code style="color: rgba(255,255,255,0.8);">MARKETING_KNOWLEDGE_BASE</code>
            </div>
            <div style="margin-bottom: 1rem;">
                <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem; text-transform: uppercase;">Content</div>
                <span style="color: rgba(255,255,255,0.8);">Campaign Briefs, Creative Strategy Decks, Competitor Analysis</span>
            </div>
            <div>
                <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem; text-transform: uppercase;">Indexed By</div>
                <span style="color: rgba(255,255,255,0.8);">Campaign ID, Product Category, Region</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Sample Questions:**")
        st.markdown(f"""
        <div style="background: {BG_HOVER}; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
            <div style="color: rgba(255,255,255,0.9); font-style: italic;">
                "What was the messaging hook for the Science of Safety campaign?"
            </div>
        </div>
        <div style="background: {BG_HOVER}; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
            <div style="color: rgba(255,255,255,0.9); font-style: italic;">
                "Summarize creative strategy for top LinkedIn campaigns"
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================================================================
    # 6. APPLICATION PAGES
    # =========================================================================
    st.markdown("## Application Pages")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background: {BG_CARD}; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;">
            <h4 style="color: {COLOR_PRIMARY}; margin-bottom: 0.75rem;">Strategic Dashboard</h4>
            <p style="color: rgba(255,255,255,0.7); margin-bottom: 0.5rem;">
                Unified view of ROI across all business units with revenue attribution, 
                regional heatmaps, and actionable recommendations.
            </p>
            <p style="color: rgba(255,255,255,0.5); font-size: 0.85rem;">
                <em>Persona: CMO / VP Marketing</em>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background: {BG_CARD}; border-radius: 12px; padding: 1.5rem;">
            <h4 style="color: {COLOR_PRIMARY}; margin-bottom: 0.75rem;">Model Explorer</h4>
            <p style="color: rgba(255,255,255,0.7); margin-bottom: 0.5rem;">
                Deep-dive into response curves, confidence intervals, and ad-hoc 
                analysis via the Cortex Analyst chat interface.
            </p>
            <p style="color: rgba(255,255,255,0.5); font-size: 0.85rem;">
                <em>Persona: Data Scientist / Marketing Analyst</em>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: {BG_CARD}; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;">
            <h4 style="color: {COLOR_PRIMARY}; margin-bottom: 0.75rem;">Budget Simulator</h4>
            <p style="color: rgba(255,255,255,0.7); margin-bottom: 0.5rem;">
                "What-if" scenario modeling—adjust sliders to simulate budget reallocation 
                and see predicted revenue impact in real-time.
            </p>
            <p style="color: rgba(255,255,255,0.5); font-size: 0.85rem;">
                <em>Persona: Regional Demand Gen Lead</em>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
        st.markdown(f"""
        <div style="background: {BG_CARD}; border-radius: 12px; padding: 1.5rem;">
            <h4 style="color: {COLOR_PRIMARY}; margin-bottom: 0.75rem;">About</h4>
            <p style="color: rgba(255,255,255,0.7); margin-bottom: 0.5rem;">
                Project documentation for business and technical audiences 
                (you are here).
            </p>
            <p style="color: rgba(255,255,255,0.5); font-size: 0.85rem;">
                <em>Persona: All Users</em>
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================================================================
    # 7. TARGET BUSINESS OUTCOMES
    # =========================================================================
    st.markdown("## Target Business Outcomes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {COLOR_PRIMARY}22, {COLOR_SECONDARY}22); 
                    border: 1px solid {COLOR_PRIMARY}; border-radius: 12px; padding: 2rem; text-align: center;">
            <div style="font-size: 3rem; font-weight: 700; color: {COLOR_PRIMARY};">15%</div>
            <div style="color: white; font-size: 1.1rem; margin-top: 0.5rem;">
                Improvement in Marketing Efficiency Ratio
            </div>
            <p style="color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-top: 1rem;">
                Revenue generated per dollar of marketing spend through algorithmic budget reallocation
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {COLOR_ACCENT}22, {COLOR_ACCENT}11); 
                    border: 1px solid {COLOR_ACCENT}; border-radius: 12px; padding: 2rem; text-align: center;">
            <div style="font-size: 3rem; font-weight: 700; color: {COLOR_ACCENT};">2x</div>
            <div style="color: white; font-size: 1.1rem; margin-top: 0.5rem;">
                Faster Budget Decision Cycles
            </div>
            <p style="color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-top: 1rem;">
                From quarterly planning cycles to real-time simulation and optimization
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================================================================
    # 8. TECHNOLOGY STACK
    # =========================================================================
    st.markdown("## Technology Stack")
    
    tech_items = [
        ("Snowflake", "Data warehouse, compute, and governance"),
        ("Snowpark ML", "Python ML training on Snowflake compute"),
        ("Cortex AI", "Natural language analytics (Analyst & Search)"),
        ("Streamlit", "Interactive web application framework"),
        ("Nevergrad", "Hyperparameter optimization for MMM"),
        ("Ridge Regression", "Core MMM algorithm with positive constraints")
    ]
    
    cols = st.columns(3)
    for i, (tech, desc) in enumerate(tech_items):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background: {BG_CARD}; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <div style="color: white; font-weight: 600;">{tech}</div>
                <div style="color: rgba(255,255,255,0.5); font-size: 0.85rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================================================================
    # 9. GETTING STARTED
    # =========================================================================
    st.markdown("## Getting Started")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Quick Start Guide")
        st.markdown(f"""
        <div style="background: {BG_CARD}; border-radius: 12px; padding: 1.5rem;">
            <ol style="color: rgba(255,255,255,0.85); line-height: 2; padding-left: 1.25rem; margin: 0;">
                <li>Start with the <strong>Strategic Dashboard</strong> for an executive overview</li>
                <li>Use filters to drill down by region, product, or channel</li>
                <li>Go to <strong>Budget Simulator</strong> to test "what-if" scenarios</li>
                <li>Explore <strong>Model Explorer</strong> for detailed response curves</li>
                <li>Ask questions in the Cortex Analyst chat interface</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Interpreting Results")
        st.markdown(f"""
        <div style="background: {BG_CARD}; border-radius: 12px; padding: 1.5rem;">
            <div style="margin-bottom: 1rem;">
                <div style="color: {COLOR_PRIMARY}; font-weight: 600;">ROI with Confidence Interval</div>
                <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">
                    "3.2x [2.8 - 3.6]" means we're 90% confident the true ROI is between 2.8x and 3.6x.
                </div>
            </div>
            <div style="margin-bottom: 1rem;">
                <div style="color: {COLOR_SUCCESS}; font-weight: 600;">Significant</div>
                <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">
                    The entire CI is above zero—statistically reliable.
                </div>
            </div>
            <div>
                <div style="color: {COLOR_ACCENT}; font-weight: 600;">Marginal ROI</div>
                <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">
                    Return on the <em>next</em> dollar—use this for reallocation decisions.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================================================================
    # 10. THE "WOW" MOMENT
    # =========================================================================
    st.markdown("## The Demo \"Wow\" Moment")
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {BG_CARD}, #1e2738); 
                border-radius: 12px; padding: 2rem; margin: 1rem 0; border: 1px solid rgba(41, 181, 232, 0.3);">
        <p style="color: rgba(255,255,255,0.7); font-style: italic; font-size: 1.1rem;">
            The CMO asks the Cortex Analyst:
        </p>
        <p style="color: white; font-size: 1.2rem; margin: 1rem 0; padding-left: 1rem; border-left: 3px solid {COLOR_PRIMARY};">
            "How should we reallocate our Q3 budget to maximize industrial distributor sales in APAC?"
        </p>
        <p style="color: rgba(255,255,255,0.7); margin-top: 1rem;">
            The system instantly parses the underlying MMM results and generates:
        </p>
        <p style="color: {COLOR_ACCENT}; font-size: 1.1rem; font-weight: 600; margin-top: 0.5rem;">
            "Shift $500K from Programmatic Display to LinkedIn Video for a projected $2.4M revenue lift"
        </p>
        <p style="color: rgba(255,255,255,0.5); font-size: 0.9rem; margin-top: 1rem;">
            — displayed alongside an interactive saturation curve showing exactly where each channel sits on its diminishing returns curve
        </p>
    </div>
    """, unsafe_allow_html=True)

    # =========================================================================
    # FOOTER
    # =========================================================================
    st.divider()
    
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem; color: rgba(255,255,255,0.4);">
        <p style="font-size: 0.85rem;">
            Built with Snowflake • Snowpark ML • Cortex AI • Streamlit
        </p>
        <p style="font-size: 0.75rem; margin-top: 0.5rem;">
            Model Version: v3.0_hierarchical | Data refreshed weekly
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
