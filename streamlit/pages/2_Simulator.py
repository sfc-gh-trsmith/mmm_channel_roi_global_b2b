"""
Budget Simulator - Regional Demand Gen Lead Persona

"Flight Simulator" interface for what-if budget allocation scenarios.
Allows users to adjust spend per channel and see predicted revenue impact.

ENHANCED FEATURES:
- Saturation zone annotations on response curves
- Efficiency score for portfolio
- CI bands on predictions
- Adstock decay visualization per channel
- Quick action presets (Optimize, Maximize, Balance)
- Educational panels on MMM concepts
"""
import streamlit as st
import pandas as pd
import numpy as np
import math
import logging
import plotly.graph_objects as go
from snowflake.snowpark.context import get_active_session
import sys
from pathlib import Path

DEPLOY_VERSION = "321D1792-4D99-4D0C-B1B0-727D83BE06CF"

logger = logging.getLogger("snowflake.connector")
logger.setLevel(logging.INFO)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import run_queries_parallel
from utils.styling import (
    inject_custom_css,
    render_story_section,
    render_learn_more_panel,
    render_zone_badge,
    render_adstock_decay_viz,
    render_parameter_pill,
    add_saturation_zone_annotation,
    add_confidence_band_to_line_chart,
    apply_plotly_theme,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_ACCENT,
    COLOR_SUCCESS,
    COLOR_DANGER,
    COLOR_WARNING,
    BG_CARD
)
from utils.explanations import get_explanation

# --- Page Config ---
st.set_page_config(
    page_title="Budget Simulator | MMM ROI Engine",
    layout="wide"
)

inject_custom_css()


def log_event(session, event_type: str, message: str, details: dict = None):
    """Log debug event to Snowflake event table."""
    try:
        import json
        details_str = json.dumps(details) if details else "{}"
        logger.info(f"[SIMULATOR][{DEPLOY_VERSION[:8]}] {event_type}: {message} | {details_str}")
    except Exception as e:
        pass


@st.cache_data(ttl=300)
def load_simulator_data(_session):
    """Load response curves and model results for simulation (with enhanced fields)."""
    from utils.data_loader import QUERIES, DATABASE
    
    queries = {
        "CURVES": QUERIES["CURVES"],
        "RESULTS": QUERIES["RESULTS"],
        "WEEKLY": QUERIES["WEEKLY"],
    }
    
    return run_queries_parallel(_session, queries)


def interpolate_revenue(df_curve: pd.DataFrame, spend: float, 
                         return_ci: bool = False) -> tuple:
    """
    Interpolate predicted revenue from response curve.
    Optionally return CI bounds if available.
    
    Note: Floors revenue at 0 since negative revenue predictions indicate
    model uncertainty rather than actual negative returns.
    """
    if df_curve.empty:
        if return_ci:
            return 0.0, 0.0, 0.0
        return 0.0
    
    df_sorted = df_curve.sort_values('SPEND')
    revenue = max(0.0, float(np.interp(spend, df_sorted['SPEND'], df_sorted['PREDICTED_REVENUE'])))
    
    if return_ci:
        if 'PREDICTED_REVENUE_CI_LOWER' in df_sorted.columns:
            ci_lower = max(0.0, float(np.interp(spend, df_sorted['SPEND'], 
                                       df_sorted['PREDICTED_REVENUE_CI_LOWER'].fillna(revenue * 0.85))))
            ci_upper = max(0.0, float(np.interp(spend, df_sorted['SPEND'], 
                                       df_sorted['PREDICTED_REVENUE_CI_UPPER'].fillna(revenue * 1.15))))
        else:
            ci_lower = max(0.0, revenue * 0.85)
            ci_upper = max(0.0, revenue * 1.15)
        return revenue, ci_lower, ci_upper
    
    return revenue


def calculate_efficiency_score(channel_impacts: list, channel_params: dict) -> dict:
    """
    Calculate portfolio efficiency score based on ACTUAL marginal ROI at simulated spend level.
    
    Uses Hill saturation parameters to compute true mROI at each spend point,
    accounting for diminishing returns as channels approach saturation.
    
    Returns:
    - score: 0-100 (100 = perfect allocation)
    - interpretation: text description
    - details: per-channel efficiency metrics
    """
    if not channel_impacts:
        return {"score": 50, "interpretation": "No data", "details": []}
    
    total_spend = sum(c['Simulated Spend'] for c in channel_impacts)
    if total_spend == 0:
        return {"score": 0, "interpretation": "No spend allocated", "details": []}
    
    weighted_mroi = 0
    details = []
    
    for c in channel_impacts:
        ch = c['Channel']
        spend = c['Simulated Spend']
        params = channel_params.get(ch, {})
        
        alpha = params.get('saturation_alpha', 2.5)
        gamma = params.get('saturation_gamma', 50000)
        base_mroi = params.get('marginal_roi', 1.0)
        weekly_spend = params.get('weekly_spend', gamma)
        
        x_alpha = spend ** alpha
        gamma_alpha = gamma ** alpha
        saturation = x_alpha / (x_alpha + gamma_alpha)
        
        if weekly_spend > 0:
            base_x_alpha = weekly_spend ** alpha
            base_saturation = base_x_alpha / (base_x_alpha + gamma_alpha)
        else:
            base_saturation = 0.5
        
        if spend > 0 and weekly_spend > 0:
            ratio = ((1 - saturation) / (1 - base_saturation)) * (weekly_spend / spend) ** (alpha - 1)
            mroi_at_spend = base_mroi * min(ratio, 2.0)
        else:
            mroi_at_spend = base_mroi
        
        weight = spend / total_spend
        weighted_mroi += mroi_at_spend * weight
        
        if saturation < 0.3:
            efficiency = "EFFICIENT"
        elif saturation < 0.7:
            efficiency = "DIMINISHING"
        else:
            efficiency = "SATURATED"
        
        details.append({
            "channel": ch,
            "spend_share": weight,
            "marginal_roi": mroi_at_spend,
            "saturation": saturation,
            "efficiency": efficiency
        })
    
    score = min(100, max(0, (weighted_mroi - 0.5) / 1.5 * 100))
    
    if score >= 80:
        interpretation = "Excellent allocation - High marginal returns across portfolio"
    elif score >= 60:
        interpretation = "Good allocation - Moderate efficiency, some optimization opportunity"
    elif score >= 40:
        interpretation = "Fair allocation - Significant budget in saturated channels"
    else:
        interpretation = "Poor allocation - Heavy spend in low-return zones"
    
    return {
        "score": score,
        "weighted_mroi": weighted_mroi,
        "interpretation": interpretation,
        "details": details
    }


def main():
    from utils.data_loader import QUERIES, DATABASE
    
    # --- Session & Data ---
    try:
        session = get_active_session()
        logger.info(f"[SIMULATOR][{DEPLOY_VERSION[:8]}] SESSION_INIT: Session acquired")
    except Exception as e:
        st.error("Could not connect to Snowflake. Please ensure you're running in Snowflake.")
        return
    
    logger.info(f"[SIMULATOR][{DEPLOY_VERSION[:8]}] MAIN_START: version={DEPLOY_VERSION}, db={DATABASE}")
    logger.info(f"[SIMULATOR][{DEPLOY_VERSION[:8]}] CURVES_QUERY: {QUERIES['CURVES']}")
    
    # Clear cache button for debugging
    if st.sidebar.button("Clear Cache & Reload"):
        st.cache_data.clear()
        st.rerun()
    
    with st.spinner("Loading simulator data..."):
        data = load_simulator_data(session)
        df_curves = data.get("CURVES", pd.DataFrame())
        df_results = data.get("RESULTS", pd.DataFrame())
        df_weekly = data.get("WEEKLY", pd.DataFrame())
    
    logger.info(f"[SIMULATOR][{DEPLOY_VERSION[:8]}] DATA_LOADED: curves_empty={df_curves.empty}, curves_rows={len(df_curves)}, results_rows={len(df_results)}, weekly_rows={len(df_weekly)}")
    
    if not df_curves.empty:
        logger.info(f"[SIMULATOR][{DEPLOY_VERSION[:8]}] CURVES_COLS: {list(df_curves.columns)}")
        logger.info(f"[SIMULATOR][{DEPLOY_VERSION[:8]}] CURVES_SAMPLE: {df_curves.head(1).to_dict()}")

    # --- Header ---
    st.markdown(f"""
    <div style="margin-bottom: 1.5rem;">
        <h1>Budget Allocation Simulator</h1>
        <p style="color: rgba(255,255,255,0.6); font-size: 1.1rem;">
            Model "what-if" scenarios to predict revenue impact with confidence intervals
        </p>
        <p style="color: rgba(255,255,255,0.3); font-size: 0.7rem;">v{DEPLOY_VERSION[:8]} | curves={len(df_curves)}</p>
    </div>
    """, unsafe_allow_html=True)

    if df_curves.empty:
        logger.info(f"[SIMULATOR][{DEPLOY_VERSION[:8]}] NO_CURVES: DataFrame is empty, showing warning")
        st.warning("No response curves found. Please run the MMM training pipeline first.")
        st.markdown(
            render_story_section(
                "Getting Started",
                "The simulator requires trained MMM response curves. "
                "Run the training notebook to generate channel-specific saturation curves."
            ),
            unsafe_allow_html=True
        )
        return

    # Process channels - extract base channel from names like "Facebook_GLOBAL_ALL"
    def extract_base_channel(channel_name):
        parts = channel_name.split('_')
        if len(parts) >= 2:
            return parts[0]  # Facebook, LinkedIn, Google Ads, Programmatic
        return channel_name
    
    df_curves['Base_Channel'] = df_curves['CHANNEL'].apply(extract_base_channel)

    # --- Scenario Context ---
    st.markdown(
        render_story_section(
            "The Question",
            "What happens if I shift budget between channels? "
            "Use the sliders below to simulate reallocation and see predicted revenue impact."
        ),
        unsafe_allow_html=True
    )

    # Use all channels (no regional filtering since data is GLOBAL)
    channels = df_curves['CHANNEL'].unique().tolist()

    if not channels:
        st.info("No channels available.")
        return

    # Get model results for parameter display (needed for quick actions)
    channel_params = {}
    if not df_results.empty:
        for _, row in df_results.iterrows():
            ch = row.get('CHANNEL', '')
            current_spend = row.get('CURRENT_SPEND', 0)
            weekly_spend = current_spend / 260 if current_spend > 0 else 0
            channel_params[ch] = {
                'adstock_decay': row.get('ADSTOCK_DECAY', 0.5),
                'saturation_alpha': row.get('SATURATION_ALPHA', 2.5),
                'saturation_gamma': row.get('SATURATION_GAMMA', 50000),
                'marginal_roi': row.get('MARGINAL_ROI', 1.0),
                'is_significant': row.get('IS_SIGNIFICANT', True),
                'weekly_spend': weekly_spend
            }
    
    # --- Reset All Button (outside expander) ---
    if st.button("Reset All", use_container_width=False, help="Reset budget and all channel allocations to baseline"):
        st.session_state['budget_adjustment'] = 0
        for key in list(st.session_state.keys()):
            if key.startswith("slider_"):
                del st.session_state[key]
        st.rerun()
    
    # --- Budget Planning Section ---
    with st.expander("Budget & Optimization", expanded=True):
        st.markdown(
            "<p style='color: rgba(255,255,255,0.7); font-size: 0.9rem; margin-bottom: 0.5rem;'>"
            "<b>Step 1:</b> Set your target budget, then <b>Step 2:</b> Click an action button</p>",
            unsafe_allow_html=True
        )
        
        # Calculate baseline total for reference
        temp_total_baseline = 0
        for ch in channels:
            params_temp = channel_params.get(ch, {})
            ch_curve = df_curves[df_curves['CHANNEL'] == ch]
            base_spend = params_temp.get('weekly_spend', 0)
            if base_spend == 0:
                base_spend = float(ch_curve['SPEND'].mean())
            temp_total_baseline += base_spend
        
        # Initialize budget adjustment in session state
        if 'budget_adjustment' not in st.session_state:
            st.session_state['budget_adjustment'] = 0
        
        # Budget slider
        col_slider, col_info = st.columns([2, 1])
        with col_slider:
            budget_pct = st.slider(
                "Budget Adjustment",
                min_value=-50,
                max_value=100,
                value=st.session_state['budget_adjustment'],
                step=5,
                format="%+d%%",
                key="budget_slider",
                help="Adjust total budget relative to baseline"
            )
            st.session_state['budget_adjustment'] = budget_pct
        
        target_budget = temp_total_baseline * (1 + budget_pct / 100)
        
        with col_info:
            change_color = COLOR_SUCCESS if budget_pct >= 0 else COLOR_DANGER
            st.markdown(
                f"<div style='background: rgba(255,255,255,0.05); padding: 0.75rem; border-radius: 8px; text-align: center;'>"
                f"<span style='color: rgba(255,255,255,0.5); font-size: 0.8rem;'>Target Budget</span><br>"
                f"<span style='font-size: 1.3rem; font-weight: 700; color: {change_color};'>${target_budget:,.0f}</span><br>"
                f"<span style='color: rgba(255,255,255,0.4); font-size: 0.75rem;'>Baseline: ${temp_total_baseline:,.0f}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        st.markdown("<div style='margin-top: 0.75rem;'></div>", unsafe_allow_html=True)
        
        # Action buttons
        qa1, qa2, qa3, qa4 = st.columns(4)
        
        with qa1:
            original_alloc_clicked = st.button("Original Allocation", use_container_width=True,
                                               help="Use baseline proportions at target budget")
        with qa2:
            optimize_clicked = st.button("Optimize ROI", use_container_width=True,
                                         help="Allocate target budget toward higher-ROI channels")
        with qa3:
            balance_clicked = st.button("Balance", use_container_width=True,
                                        help="Distribute target budget evenly across channels")
        with qa4:
            scale_up = st.button("Scale +20%", use_container_width=True,
                                 help="Increase all current spend by 20% (ignores target budget)")
    
    # Pre-process quick actions before the slider loop
    # This ensures session state is updated BEFORE sliders are rendered
    if original_alloc_clicked or optimize_clicked or balance_clicked or scale_up:
        # Need to calculate parameters first
        temp_channel_params = {}
        temp_baseline_allocation = {}
        temp_total_baseline = 0
        
        for ch in channels:
            params_temp = channel_params.get(ch, {})
            ch_curve = df_curves[df_curves['CHANNEL'] == ch]
            base_spend = params_temp.get('weekly_spend', 0)
            if base_spend == 0:
                base_spend = float(ch_curve['SPEND'].mean())
            temp_baseline_allocation[ch] = base_spend
            temp_total_baseline += base_spend
            temp_channel_params[ch] = params_temp
        
        # Get target budget from slider (default to baseline if not set)
        budget_pct = st.session_state.get('budget_adjustment', 0)
        target_total = temp_total_baseline * (1 + budget_pct / 100)
        
        if original_alloc_clicked:
            # Use BASELINE proportions, scale to target budget
            scale_factor = target_total / temp_total_baseline if temp_total_baseline > 0 else 1.0
            
            for c in channels:
                ch_curve = df_curves[df_curves['CHANNEL'] == c]
                max_spend = float(ch_curve['SPEND'].max()) * 1.5
                scaled_spend = temp_baseline_allocation[c] * scale_factor
                st.session_state[f"slider_{c}"] = max(0.0, min(scaled_spend, max_spend))
        
        elif optimize_clicked:
            saturated_channels = []
            efficient_channels = []
            high_contrib_channels = []
            
            for c in channels:
                params = temp_channel_params.get(c, {})
                spend = temp_baseline_allocation[c]
                alpha = params.get('saturation_alpha', 2.5)
                gamma = params.get('saturation_gamma', 50000)
                base_mroi = params.get('marginal_roi', 1.0)
                
                x_alpha = spend ** alpha
                gamma_alpha = gamma ** alpha
                saturation = x_alpha / (x_alpha + gamma_alpha)
                
                weight = spend / temp_total_baseline if temp_total_baseline > 0 else 0
                contrib = base_mroi * weight
                
                if saturation > 0.5:
                    saturated_channels.append(c)
                elif saturation < 0.3 and base_mroi > 0.5:
                    efficient_channels.append((c, base_mroi))
                
                if contrib > 0.1:
                    high_contrib_channels.append(c)
            
            cut_total = 0
            raw_allocations = {}
            for c in channels:
                if c in saturated_channels:
                    cut_amount = temp_baseline_allocation[c] * 0.40
                    raw_allocations[c] = temp_baseline_allocation[c] - cut_amount
                    cut_total += cut_amount
                else:
                    raw_allocations[c] = temp_baseline_allocation[c]
            
            redistribute_channels = [(ch, mroi) for ch, mroi in efficient_channels 
                                     if ch not in high_contrib_channels]
            
            if redistribute_channels and cut_total > 0:
                total_mroi = sum(mroi for _, mroi in redistribute_channels)
                for ch, mroi in redistribute_channels:
                    add_amount = cut_total * (mroi / total_mroi)
                    raw_allocations[ch] = raw_allocations[ch] + add_amount
            elif efficient_channels and cut_total > 0:
                total_mroi = sum(mroi for _, mroi in efficient_channels)
                for ch, mroi in efficient_channels:
                    add_amount = cut_total * (mroi / total_mroi)
                    raw_allocations[ch] = raw_allocations[ch] + add_amount
            
            raw_total = sum(raw_allocations.values())
            # Scale to target budget instead of baseline
            scale_factor = target_total / raw_total if raw_total > 0 else 1.0
            
            for c in channels:
                ch_curve = df_curves[df_curves['CHANNEL'] == c]
                max_spend = float(ch_curve['SPEND'].max()) * 1.5
                optimal_spend = raw_allocations[c] * scale_factor
                st.session_state[f"slider_{c}"] = max(0.0, min(optimal_spend, max_spend))
        
        elif balance_clicked:
            # Use target budget for balance
            balanced_value = target_total / len(channels) if channels else 0
            for c in channels:
                ch_curve = df_curves[df_curves['CHANNEL'] == c]
                max_spend = float(ch_curve['SPEND'].max()) * 1.5
                st.session_state[f"slider_{c}"] = min(balanced_value, max_spend)
        
        elif scale_up:
            # Scale +20% ignores target budget, just scales current values
            for c in channels:
                ch_curve = df_curves[df_curves['CHANNEL'] == c]
                max_spend = float(ch_curve['SPEND'].max()) * 1.5
                current_val = st.session_state.get(f"slider_{c}", temp_baseline_allocation[c])
                st.session_state[f"slider_{c}"] = min(current_val * 1.2, max_spend)
        
        st.rerun()

    # --- Main Layout ---
    col_sliders, col_results = st.columns([1, 2])

    # Store allocations
    current_allocation = {}
    baseline_allocation = {}
    
    # Initialize gross margin in session state
    if 'gross_margin' not in st.session_state:
        st.session_state['gross_margin'] = 0.70  # Default 70% gross margin

    with col_sliders:
        st.markdown("### Adjust Spend by Channel")
        
        # Pre-calculate total baseline for percentage calculations
        # Use actual weekly spend from MODEL_RESULTS (CURRENT_SPEND / 260 weeks)
        total_baseline = 0
        for ch in channels:
            params = channel_params.get(ch, {})
            ch_curve = df_curves[df_curves['CHANNEL'] == ch]
            # Use actual weekly spend if available, otherwise fall back to curve mean
            base_spend = params.get('weekly_spend', 0)
            if base_spend == 0:
                base_spend = float(ch_curve['SPEND'].mean())
            baseline_allocation[ch] = base_spend
            total_baseline += base_spend
        
        # Placeholder for Total Spend Summary - will be populated after sliders
        spend_summary_placeholder = st.empty()
        
        st.markdown("---")
        
        # Collapsible: Business Settings (Gross Margin, Target ROAS)
        with st.expander("Business Settings", expanded=False):
            st.markdown(
                "<p style='color: rgba(255,255,255,0.6); font-size: 0.85rem;'>"
                "Adjust these to match your business economics</p>",
                unsafe_allow_html=True
            )
            
            gross_margin = st.slider(
                "Gross Margin %",
                min_value=10,
                max_value=90,
                value=int(st.session_state['gross_margin'] * 100),
                step=5,
                help="Your product gross margin. Higher margin = lower breakeven ROAS needed."
            )
            st.session_state['gross_margin'] = gross_margin / 100
            
            # Calculate breakeven ROAS
            breakeven_roas = 1 / (gross_margin / 100) if gross_margin > 0 else float('inf')
            
            st.markdown(
                f"<div style='background: rgba(255,255,255,0.05); padding: 0.75rem; border-radius: 8px; margin-top: 0.5rem;'>"
                f"<span style='color: rgba(255,255,255,0.6);'>Breakeven ROAS:</span> "
                f"<span style='font-weight: 600; color: {COLOR_WARNING};'>{breakeven_roas:.2f}x</span><br>"
                f"<span style='color: rgba(255,255,255,0.5); font-size: 0.8rem;'>"
                f"At {gross_margin}% margin, you need ${breakeven_roas:.2f} revenue per $1 spent to break even on profit.</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        # Educational expander
        with st.expander("How to read saturation curves", expanded=False):
            exp = get_explanation("saturation_curves")
            st.markdown(exp.get("content", ""), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Reset total_simulated for accurate calculation during slider loop
        total_simulated = 0
        
        for ch in channels:
            ch_curve = df_curves[df_curves['CHANNEL'] == ch]
            params = channel_params.get(ch, {})
            
            # Get baseline values
            base_spend = baseline_allocation[ch]
            max_spend = float(ch_curve['SPEND'].max()) * 1.5
            min_spend = 0.0
            
            slider_key = f"slider_{ch}"
            
            # Quick actions are now handled before this loop with st.rerun()
            # Just initialize slider state if not already set
            
            if slider_key not in st.session_state:
                st.session_state[slider_key] = base_spend
            
            # Calculate percentage of total budget
            pct_of_total = (base_spend / total_baseline * 100) if total_baseline > 0 else 0
            
            # Get current slider value from session state for zone calculation
            current_val = st.session_state.get(slider_key, base_spend)
            
            # Calculate efficiency zone based on spend relative to gamma (matches chart zones)
            gamma = params.get('saturation_gamma', 50000)
            if current_val < gamma * 0.5:
                zone = "EFFICIENT"
            elif current_val < gamma * 1.5:
                zone = "DIMINISHING"
            else:
                zone = "SATURATED"
            
            col_ch, col_zone = st.columns([3, 1])
            with col_ch:
                st.markdown(
                    f"**{ch}** <span style='color: {COLOR_PRIMARY}; font-size: 0.85rem;'>"
                    f"({pct_of_total:.1f}% • ${base_spend:,.0f})</span>",
                    unsafe_allow_html=True
                )
            with col_zone:
                st.markdown(render_zone_badge(zone), unsafe_allow_html=True)
            
            # Note: Don't pass value= when using key= with session state to avoid warning
            val = st.slider(
                ch,
                min_value=min_spend,
                max_value=max_spend,
                format="$%.0f",
                key=slider_key,
                label_visibility="collapsed"
            )
            
            # Display formatted value with commas prominently
            st.markdown(
                f"<div style='text-align: center; margin-top: -0.3rem; margin-bottom: 0.5rem;'>"
                f"<span style='font-size: 1.2rem; font-weight: 700; color: #4CAF50; background: rgba(76,175,80,0.1); "
                f"padding: 0.2rem 0.8rem; border-radius: 4px;'>${val:,.0f}</span></div>",
                unsafe_allow_html=True
            )
            
            current_allocation[ch] = val
            total_simulated += val
            
            # Recalculate zone based on actual slider value (for display updates)
            if val < gamma * 0.5:
                zone = "EFFICIENT"
            elif val < gamma * 1.5:
                zone = "DIMINISHING"
            else:
                zone = "SATURATED"
            
            # Show delta in both $ and % formats
            delta = val - base_spend
            pct_delta = ((val - base_spend) / base_spend * 100) if base_spend > 0 else 0
            
            col_delta, col_adstock = st.columns([2, 1])
            
            with col_delta:
                if delta != 0:
                    delta_color = COLOR_SUCCESS if delta > 0 else COLOR_DANGER
                    delta_text = f"+${delta:,.0f}" if delta > 0 else f"-${abs(delta):,.0f}"
                    pct_text = f"({pct_delta:+.1f}%)"
                    st.markdown(
                        f"<span style='color: {delta_color}; font-size: 0.8rem;'>"
                        f"{delta_text} {pct_text}</span>",
                        unsafe_allow_html=True
                    )
            
            with col_adstock:
                if params.get('adstock_decay'):
                    st.markdown(
                        f"<span title='Adstock decay rate' style='font-size: 0.7rem; color: rgba(255,255,255,0.5);'>"
                        f"θ={params['adstock_decay']:.2f}</span>",
                        unsafe_allow_html=True
                    )
            
            st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
        
        # Now populate Total Spend Summary with accurate values (after all sliders processed)
        with spend_summary_placeholder.container():
            spend_delta = total_simulated - total_baseline
            spend_pct_delta = ((total_simulated - total_baseline) / total_baseline * 100) if total_baseline > 0 else 0
            
            col_spend1, col_spend2 = st.columns(2)
            with col_spend1:
                st.metric(
                    "Simulated Spend",
                    f"${total_simulated:,.0f}",
                    delta=f"{spend_pct_delta:+.1f}%" if abs(spend_delta) > 1 else None,
                    delta_color="inverse"
                )
            with col_spend2:
                st.metric(
                    "Baseline Spend",
                    f"${total_baseline:,.0f}",
                    help="Weekly average based on historical spend"
                )
        
        # Educational: Adstock
        with st.expander("Learn More: How Adstock Works", expanded=False):
            exp = get_explanation("adstock_carryover")
            st.markdown(exp.get("content", ""), unsafe_allow_html=True)

    with col_results:
        st.markdown("### Simulation Results")
        
        # Calculate revenue predictions with CI
        baseline_revenue = 0
        simulated_revenue = 0
        simulated_ci_lower = 0
        simulated_ci_upper = 0
        channel_impacts = []
        
        for ch in channels:
            ch_curve = df_curves[df_curves['CHANNEL'] == ch].sort_values('SPEND')
            
            base_spend = baseline_allocation[ch]
            sim_spend = current_allocation[ch]
            
            base_rev = interpolate_revenue(ch_curve, base_spend)
            sim_rev, sim_ci_l, sim_ci_u = interpolate_revenue(ch_curve, sim_spend, return_ci=True)
            
            baseline_revenue += base_rev
            simulated_revenue += sim_rev
            simulated_ci_lower += sim_ci_l
            simulated_ci_upper += sim_ci_u
            
            # Get zone at simulated spend
            if 'EFFICIENCY_ZONE' in ch_curve.columns:
                zone_at_spend = ch_curve[ch_curve['SPEND'] <= sim_spend]['EFFICIENCY_ZONE'].iloc[-1] \
                    if len(ch_curve[ch_curve['SPEND'] <= sim_spend]) > 0 else 'UNKNOWN'
            else:
                gamma = channel_params.get(ch, {}).get('saturation_gamma', 50000)
                if sim_spend < gamma * 0.5:
                    zone_at_spend = "EFFICIENT"
                elif sim_spend < gamma * 1.5:
                    zone_at_spend = "DIMINISHING"
                else:
                    zone_at_spend = "SATURATED"
            
            channel_impacts.append({
                'Channel': ch,
                'Baseline Spend': base_spend,
                'Simulated Spend': sim_spend,
                'Spend Delta': sim_spend - base_spend,
                'Baseline Revenue': base_rev,
                'Simulated Revenue': sim_rev,
                'Revenue Delta': sim_rev - base_rev,
                'CI_Lower': sim_ci_l,
                'CI_Upper': sim_ci_u,
                'Zone': zone_at_spend
            })
        
        # Revenue impact metrics with CI
        rev_delta = simulated_revenue - baseline_revenue
        rev_pct_delta = ((simulated_revenue - baseline_revenue) / baseline_revenue * 100) if baseline_revenue > 0 else 0
        
        def format_currency_short(val):
            if abs(val) >= 1_000_000:
                return f"${val/1_000_000:.2f}M"
            elif abs(val) >= 1_000:
                return f"${val/1_000:.1f}K"
            return f"${val:,.0f}"
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric(
                "Simulated Revenue",
                format_currency_short(simulated_revenue),
                delta=f"{rev_pct_delta:+.1f}%" if rev_delta != 0 else "No change",
                delta_color="normal" if rev_delta >= 0 else "inverse"
            )
        with col_m2:
            st.metric(
                "90% CI Range",
                f"{format_currency_short(simulated_ci_lower)} - {format_currency_short(simulated_ci_upper)}",
                help="Statistical confidence interval for simulated revenue"
            )
        with col_m3:
            # Calculate and display ROAS
            sim_roas = simulated_revenue / total_simulated if total_simulated > 0 else 0
            breakeven_roas = 1 / st.session_state.get('gross_margin', 0.70)
            roas_color = COLOR_SUCCESS if sim_roas >= breakeven_roas else COLOR_DANGER
            st.metric(
                "Simulated ROAS",
                f"{sim_roas:.2f}x",
                delta=f"Breakeven: {breakeven_roas:.2f}x",
                delta_color="off"
            )
            st.markdown(
                f"<p style='color: {roas_color}; font-size: 0.7rem; margin-top: -0.5rem;'>"
                f"{'Above' if sim_roas >= breakeven_roas else 'Below'} breakeven</p>",
                unsafe_allow_html=True
            )
        
        # Efficiency Score
        efficiency = calculate_efficiency_score(channel_impacts, channel_params)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_eff, col_exp = st.columns([3, 1])
        with col_eff:
            st.markdown("##### Portfolio Efficiency Score")
        with col_exp:
            pass
        
        score = efficiency['score']
        score_color = COLOR_SUCCESS if score >= 70 else COLOR_WARNING if score >= 40 else COLOR_DANGER
        
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem;">
            <div style="font-size: 2.5rem; font-weight: 700; color: {score_color};">{score:.0f}</div>
            <div style="flex: 1;">
                <div style="background: rgba(255,255,255,0.1); border-radius: 4px; height: 8px; overflow: hidden;">
                    <div style="background: {score_color}; width: {score}%; height: 100%;"></div>
                </div>
                <div style="color: rgba(255,255,255,0.6); font-size: 0.85rem; margin-top: 0.5rem;">
                    {efficiency['interpretation']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("What is the Efficiency Score?", expanded=False):
            st.markdown(f"""
**The Efficiency Score (0-100)** measures how well your budget allocation maximizes marginal returns.

**How it works:**
- Calculates the **weighted average Marginal ROI** across all channels
- Weights are based on each channel's share of total spend
- Score of 100 = weighted mROI of 2.0+ (excellent)
- Score of 0 = weighted mROI of 0.5 or below (poor)

**Current Portfolio:**
- Weighted Marginal ROI: **{efficiency.get('weighted_mroi', 1.0):.2f}x**
- This means each additional $1 spent returns ~${efficiency.get('weighted_mroi', 1.0):.2f} in revenue

**Score Interpretation:**
- **70-100 (Green)**: Excellent - High marginal returns, room to grow
- **40-69 (Yellow)**: Good - Some optimization opportunity
- **0-39 (Red)**: Poor - Heavy spend in saturated channels

**To improve your score:**
1. Shift budget from SATURATED → EFFICIENT channels
2. Use "Optimize ROI" to auto-target optimal spend levels
3. Focus on channels with mROI > 1.0
            """)

        # Before/After comparison chart
        st.markdown("##### Spend Comparison: Baseline vs Simulated")
        
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            name='Baseline',
            x=channels,
            y=[baseline_allocation[c] for c in channels],
            marker_color=COLOR_PRIMARY,
            opacity=0.7
        ))
        fig_compare.add_trace(go.Bar(
            name='Simulated',
            x=channels,
            y=[current_allocation[c] for c in channels],
            marker_color=COLOR_ACCENT
        ))
        
        fig_compare = apply_plotly_theme(fig_compare)
        fig_compare.update_layout(
            title="",  # Explicit empty title to prevent "undefined"
            barmode='group',
            xaxis_title="Channel",
            yaxis_title="Spend ($)",
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=0, r=0, t=30, b=40),
            height=280
        )
        st.plotly_chart(fig_compare, use_container_width=True, key="sim_compare")

        # Response curve visualization with zones
        st.markdown("##### Response Curves with Efficiency Zones")
        
        selected_channel = st.selectbox(
            "View response curve for:",
            channels,
            key="sim_curve_select"
        )
        
        ch_curve = df_curves[df_curves['CHANNEL'] == selected_channel].sort_values('SPEND')
        ch_params = channel_params.get(selected_channel, {})
        
        if not ch_curve.empty:
            fig_curve = go.Figure()
            
            # Add CI band if available
            if 'PREDICTED_REVENUE_CI_LOWER' in ch_curve.columns:
                fig_curve = add_confidence_band_to_line_chart(
                    fig_curve,
                    ch_curve['SPEND'].tolist(),
                    ch_curve['PREDICTED_REVENUE_CI_LOWER'].fillna(ch_curve['PREDICTED_REVENUE'] * 0.85).tolist(),
                    ch_curve['PREDICTED_REVENUE_CI_UPPER'].fillna(ch_curve['PREDICTED_REVENUE'] * 1.15).tolist()
                )
            
            # Response curve line
            fig_curve.add_trace(go.Scatter(
                x=ch_curve['SPEND'],
                y=ch_curve['PREDICTED_REVENUE'],
                mode='lines',
                name='Response Curve',
                line=dict(color=COLOR_PRIMARY, width=3)
            ))
            
            # Baseline point
            base_spend = baseline_allocation[selected_channel]
            base_rev = interpolate_revenue(ch_curve, base_spend)
            fig_curve.add_trace(go.Scatter(
                x=[base_spend],
                y=[base_rev],
                mode='markers',
                name='Baseline',
                marker=dict(color='white', size=12, line=dict(color=COLOR_PRIMARY, width=2))
            ))
            
            # Simulated point
            sim_spend = current_allocation[selected_channel]
            sim_rev = interpolate_revenue(ch_curve, sim_spend)
            fig_curve.add_trace(go.Scatter(
                x=[sim_spend],
                y=[sim_rev],
                mode='markers',
                name='Simulated',
                marker=dict(color=COLOR_ACCENT, size=14, symbol='star')
            ))
            
            fig_curve = apply_plotly_theme(fig_curve)
            
            # Add saturation zone annotation if gamma is available
            gamma = ch_params.get('saturation_gamma', ch_curve['SPEND'].median())
            alpha = ch_params.get('saturation_alpha', 2.5)
            max_spend = ch_curve['SPEND'].max()
            fig_curve = add_saturation_zone_annotation(fig_curve, gamma, max_spend, alpha=alpha)
            
            fig_curve.update_layout(
                title="",  # Explicit empty title to prevent "undefined"
                xaxis_title="Spend ($)",
                yaxis_title="Predicted Revenue ($)",
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
                margin=dict(l=0, r=0, t=30, b=40),
                height=350
            )
            st.plotly_chart(fig_curve, use_container_width=True, key="sim_response_curve")
            
            # Show parameters for selected channel
            if ch_params:
                st.markdown(
                    f"<div style='display: flex; gap: 0.5rem; flex-wrap: wrap;'>"
                    f"{render_parameter_pill('θ (decay)', ch_params.get('adstock_decay', 0), '.2f')}"
                    f"{render_parameter_pill('γ (half-sat)', ch_params.get('saturation_gamma', 0), ',.0f')}"
                    f"{render_parameter_pill('mROI', ch_params.get('marginal_roi', 0), '.2f')}"
                    f"</div>",
                    unsafe_allow_html=True
                )

    # --- Impact Summary Table ---
    st.markdown("### Channel Impact Summary")
    
    df_impact = pd.DataFrame(channel_impacts)
    
    # Format for display
    df_display = df_impact.copy()
    df_display['Baseline Spend'] = df_display['Baseline Spend'].apply(lambda x: f"${x:,.0f}")
    df_display['Simulated Spend'] = df_display['Simulated Spend'].apply(lambda x: f"${x:,.0f}")
    df_display['Spend Delta'] = df_display['Spend Delta'].apply(
        lambda x: f"+${x:,.0f}" if x >= 0 else f"-${abs(x):,.0f}"
    )
    df_display['Baseline Revenue'] = df_display['Baseline Revenue'].apply(lambda x: f"${x:,.0f}")
    df_display['Simulated Revenue'] = df_display['Simulated Revenue'].apply(lambda x: f"${x:,.0f}")
    df_display['Revenue Delta'] = df_display['Revenue Delta'].apply(
        lambda x: f"+${x:,.0f}" if x >= 0 else f"-${abs(x):,.0f}"
    )
    df_display['90% CI'] = df_impact.apply(
        lambda r: f"[${r['CI_Lower']:,.0f} - ${r['CI_Upper']:,.0f}]", axis=1
    )
    
    # Select columns for display
    display_cols = ['Channel', 'Baseline Spend', 'Simulated Spend', 'Spend Delta', 
                    'Simulated Revenue', '90% CI', 'Revenue Delta', 'Zone']
    df_display = df_display[[c for c in display_cols if c in df_display.columns]]
    
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Channel": st.column_config.TextColumn("Channel", width="medium"),
            "Spend Delta": st.column_config.TextColumn("Spend Δ", width="small"),
            "Revenue Delta": st.column_config.TextColumn("Revenue Δ", width="small"),
            "Zone": st.column_config.TextColumn("Efficiency", width="small")
        }
    )
    
    # Educational: Marginal ROI
    with st.expander("Learn More: What is Marginal ROI?", expanded=False):
        exp = get_explanation("marginal_vs_average_roi")
        st.markdown(exp.get("content", ""), unsafe_allow_html=True)

    # --- Navigation ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Back to Home", use_container_width=True):
            st.switch_page("mmm_roi_app.py")
    with col2:
        if st.button("Strategic Dashboard", use_container_width=True):
            st.switch_page("pages/1_Strategic_Dashboard.py")
    with col3:
        if st.button("Model Explorer", use_container_width=True):
            st.switch_page("pages/3_Model_Explorer.py")


if __name__ == "__main__":
    main()
