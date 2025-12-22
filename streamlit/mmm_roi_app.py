"""
Global B2B Marketing Mix & ROI Engine
Hub Landing Page - Persona-Based Routing

This is the "Grand Central Station" of the application.
Users select their persona to navigate to the appropriate workflow.
"""
import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader import run_queries_parallel
from utils.styling import (
    inject_custom_css,
    render_persona_card,
    render_quick_stats,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_ACCENT
)

# --- Page Config ---
st.set_page_config(
    page_title="Global B2B MMM & ROI Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject brand styling
inject_custom_css()


@st.cache_data(ttl=3600)
def load_summary_stats(_session):
    """Load quick summary stats for the landing page."""
    # Import centralized queries from data_loader
    from utils.data_loader import QUERIES
    
    queries = {
        "ROI": QUERIES["ROI"],
        "CHANNELS": "SELECT COUNT(DISTINCT CHANNEL_CODE) as CNT FROM DIMENSIONAL.V_MMM_INPUT_WEEKLY"
    }
    return run_queries_parallel(_session, queries)


def main():
    # --- Session ---
    try:
        session = get_active_session()
        data = load_summary_stats(session)
        df_roi = data.get("ROI", pd.DataFrame())
        df_channels = data.get("CHANNELS", pd.DataFrame())
        
        # Calculate quick stats
        total_spend = df_roi['TOTAL_SPEND'].sum() if not df_roi.empty else 0
        total_rev = df_roi['ATTRIBUTED_REVENUE'].sum() if not df_roi.empty else 0
        num_channels = int(df_channels['CNT'].iloc[0]) if not df_channels.empty else 0
        has_data = True
    except Exception:
        total_spend = 0
        total_rev = 0
        num_channels = 0
        has_data = False

    # --- Hero Section ---
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem 0;">
        <h1 style="font-size: 2.8rem; margin-bottom: 0.5rem;">
            Global B2B Marketing Mix & ROI Engine
        </h1>
        <p style="font-size: 1.2rem; color: rgba(255,255,255,0.6); max-width: 600px; margin: 0 auto;">
            Unlock data-driven marketing allocation with advanced MMM 
            powered by Snowflake and Cortex AI
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Persona Cards ---
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        st.markdown(
            render_persona_card(
                title="Strategic View",
                subtitle="CMO / VP Global Markets",
                description="Executive ROI dashboard with unified channel attribution and revenue impact analysis."
            ),
            unsafe_allow_html=True
        )
        if st.button("Open Dashboard", key="btn_strategic", use_container_width=True):
            st.switch_page("pages/1_Strategic_Dashboard.py")
    
    with col2:
        st.markdown(
            render_persona_card(
                title="Operational View",
                subtitle="Regional Demand Gen Lead",
                description="What-if budget simulator to model spend reallocation scenarios and predict revenue impact."
            ),
            unsafe_allow_html=True
        )
        if st.button("Open Simulator", key="btn_operational", use_container_width=True):
            st.switch_page("pages/2_Simulator.py")
    
    with col3:
        st.markdown(
            render_persona_card(
                title="Technical View",
                subtitle="Data Scientist",
                description="Deep-dive into model diagnostics, response curves, and ad-hoc analysis with Cortex Analyst."
            ),
            unsafe_allow_html=True
        )
        if st.button("Open Explorer", key="btn_technical", use_container_width=True):
            st.switch_page("pages/3_Model_Explorer.py")

    # --- Quick Stats (ambient context) ---
    if has_data:
        st.markdown(
            render_quick_stats([
                {"value": f"${total_spend/1e6:.1f}M", "label": "Historical Spend"},
                {"value": f"${total_rev/1e6:.1f}M", "label": "Attributed Revenue"},
                {"value": str(num_channels), "label": "Active Channels"}
            ]),
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            render_quick_stats([
                {"value": "—", "label": "Historical Spend"},
                {"value": "—", "label": "Attributed Revenue"},
                {"value": "—", "label": "Active Channels"}
            ]),
            unsafe_allow_html=True
        )
        st.info("Connect to Snowflake to load live data.")

    # --- Footer ---
    st.markdown("""
    <div style="text-align: center; margin-top: 4rem; padding: 2rem; border-top: 1px solid rgba(255,255,255,0.1);">
        <p style="color: rgba(255,255,255,0.4); font-size: 0.85rem;">
            Powered by Snowflake Snowpark ML • Cortex AI • Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
