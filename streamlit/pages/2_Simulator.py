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
import plotly.graph_objects as go
from snowflake.snowpark.context import get_active_session
import sys
from pathlib import Path

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
from utils.map_viz import (
    render_region_selector_map,
    render_region_drill_down,
    render_regional_summary_metrics,
    aggregate_by_region,
    extract_region_from_channel,
    REGION_COORDS
)

# --- Page Config ---
st.set_page_config(
    page_title="Budget Simulator | MMM ROI Engine",
    layout="wide"
)

inject_custom_css()


@st.cache_data(ttl=3600)
def load_simulator_data(_session):
    """Load response curves and model results for simulation (with enhanced fields)."""
    # Import centralized queries from data_loader
    from utils.data_loader import QUERIES
    
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


def calculate_efficiency_score(channel_impacts: list, df_results: pd.DataFrame) -> dict:
    """
    Calculate portfolio efficiency score based on marginal ROI utilization.
    
    Returns:
    - score: 0-100 (100 = perfect allocation)
    - interpretation: text description
    - details: per-channel efficiency metrics
    """
    if not channel_impacts:
        return {"score": 50, "interpretation": "No data", "details": []}
    
    # Get marginal ROI data if available
    marginal_data = {}
    if not df_results.empty and 'CHANNEL' in df_results.columns and 'MARGINAL_ROI' in df_results.columns:
        for _, row in df_results.iterrows():
            marginal_data[row['CHANNEL']] = row['MARGINAL_ROI']
    
    total_spend = sum(c['Simulated Spend'] for c in channel_impacts)
    if total_spend == 0:
        return {"score": 0, "interpretation": "No spend allocated", "details": []}
    
    # Calculate weighted average marginal ROI
    weighted_mroi = 0
    details = []
    
    for c in channel_impacts:
        ch = c['Channel']
        spend = c['Simulated Spend']
        mroi = marginal_data.get(ch, 1.0)  # Default to 1.0 if not available
        weight = spend / total_spend
        weighted_mroi += mroi * weight
        
        # Classify efficiency
        if mroi > 1.5:
            efficiency = "EFFICIENT"
        elif mroi >= 0.8:
            efficiency = "DIMINISHING"
        else:
            efficiency = "SATURATED"
        
        details.append({
            "channel": ch,
            "spend_share": weight,
            "marginal_roi": mroi,
            "efficiency": efficiency
        })
    
    # Score: 100 if weighted mROI is 2.0+, 0 if 0.5 or below
    score = min(100, max(0, (weighted_mroi - 0.5) / 1.5 * 100))
    
    # Interpretation
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
    # --- Session & Data ---
    try:
        session = get_active_session()
    except Exception:
        st.error("Could not connect to Snowflake. Please ensure you're running in Snowflake.")
        return
    
    with st.spinner("Loading simulator data..."):
        data = load_simulator_data(session)
        df_curves = data.get("CURVES", pd.DataFrame())
        df_results = data.get("RESULTS", pd.DataFrame())
        df_weekly = data.get("WEEKLY", pd.DataFrame())

    # --- Header ---
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1>Budget Allocation Simulator</h1>
        <p style="color: rgba(255,255,255,0.6); font-size: 1.1rem;">
            Model "what-if" scenarios to predict revenue impact with confidence intervals
        </p>
    </div>
    """, unsafe_allow_html=True)

    if df_curves.empty:
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

    # Process channels - extract region from channel names like "Facebook_NA_ALL"
    # Region is the second-to-last segment (NA, EMEA, APAC, LATAM)
    def extract_region(channel_name):
        parts = channel_name.split('_')
        if len(parts) >= 3:
            return parts[-2]  # NA, EMEA, APAC, LATAM
        elif len(parts) == 2:
            return parts[-1]
        return 'Global'
    
    def extract_base_channel(channel_name):
        parts = channel_name.split('_')
        if len(parts) >= 2:
            return parts[0]  # Facebook, LinkedIn, Google Ads, Programmatic
        return channel_name
    
    df_curves['Region'] = df_curves['CHANNEL'].apply(extract_region)
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

    # --- Regional Map Selection ---
    st.markdown("### Select Region to Simulate")
    st.markdown(
        "<p style='color: rgba(255,255,255,0.6);'>Click a region to focus on its channels, or select All for the full portfolio</p>",
        unsafe_allow_html=True
    )
    
    # Use map for region selection if we have model results
    if not df_results.empty:
        selected_region = render_region_selector_map(df_results, key_prefix="simulator")
        
        # Show regional summary
        render_regional_summary_metrics(df_results, selected_region)
    else:
        # Fallback to dropdown if no model results
        all_regions = ['All Regions'] + sorted(df_curves['Region'].unique().tolist())
        region_selection = st.selectbox("Filter by Region", all_regions, key="sim_region")
        selected_region = None if region_selection == 'All Regions' else region_selection
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- Quick Actions ---
    col_reset, col_actions = st.columns([1, 3])
    
    with col_reset:
        if st.button("Reset to Baseline", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith("slider_"):
                    del st.session_state[key]
            st.rerun()
    
    with col_actions:
        st.markdown("<p style='font-size: 0.85rem; color: rgba(255,255,255,0.5); margin-bottom: 0.5rem;'>Quick Actions</p>", unsafe_allow_html=True)
        qa1, qa2, qa3 = st.columns(3)
        with qa1:
            optimize_clicked = st.button("Optimize ROI", use_container_width=True, 
                                         help="Shift budget to channels with highest marginal ROI")
        with qa2:
            balance_clicked = st.button("Balance", use_container_width=True,
                                        help="Distribute spend evenly across channels")
        with qa3:
            scale_up = st.button("Scale +20%", use_container_width=True,
                                 help="Increase all channel spend by 20%")

    # Filter data by region
    if selected_region:
        df_filtered = df_curves[df_curves['Region'] == selected_region]
        region_name = REGION_COORDS.get(selected_region, {}).get('full_name', selected_region)
        st.markdown(f"<h4 style='color: {COLOR_PRIMARY};'>Simulating {region_name} Channels</h4>", unsafe_allow_html=True)
    else:
        df_filtered = df_curves
    
    channels = df_filtered['CHANNEL'].unique().tolist()

    if not channels:
        st.info("No channels available for the selected region.")
        return

    # Get model results for parameter display
    channel_params = {}
    if not df_results.empty:
        for _, row in df_results.iterrows():
            ch = row.get('CHANNEL', '')
            channel_params[ch] = {
                'adstock_decay': row.get('ADSTOCK_DECAY', 0.5),
                'saturation_gamma': row.get('SATURATION_GAMMA', 50000),
                'marginal_roi': row.get('MARGINAL_ROI', 1.0),
                'is_significant': row.get('IS_SIGNIFICANT', True)
            }

    # --- Main Layout ---
    col_sliders, col_results = st.columns([1, 2])

    # Store allocations
    current_allocation = {}
    baseline_allocation = {}

    with col_sliders:
        st.markdown("### Adjust Spend by Channel")
        st.markdown(
            "<p style='color: rgba(255,255,255,0.5); font-size: 0.85rem;'>"
            "Showing % of total budget and $ amount</p>",
            unsafe_allow_html=True
        )
        
        # Educational expander
        with st.expander("How to read saturation curves", expanded=False):
            exp = get_explanation("saturation_curves")
            st.markdown(exp.get("content", ""), unsafe_allow_html=True)
        
        # Pre-calculate total baseline for percentage calculations
        total_baseline = 0
        for ch in channels:
            ch_curve = df_curves[df_curves['CHANNEL'] == ch]
            base_spend = float(ch_curve['SPEND'].mean())
            baseline_allocation[ch] = base_spend
            total_baseline += base_spend
        
        total_simulated = 0
        
        for ch in channels:
            ch_curve = df_curves[df_curves['CHANNEL'] == ch]
            params = channel_params.get(ch, {})
            
            # Get baseline values
            base_spend = baseline_allocation[ch]
            max_spend = float(ch_curve['SPEND'].max()) * 1.5
            min_spend = 0.0
            
            slider_key = f"slider_{ch}"
            
            # Handle quick actions
            if optimize_clicked and params.get('marginal_roi', 1.0) > 1.5:
                st.session_state[slider_key] = min(base_spend * 1.3, max_spend)
            elif optimize_clicked and params.get('marginal_roi', 1.0) < 0.8:
                st.session_state[slider_key] = base_spend * 0.7
            elif balance_clicked:
                balanced_value = total_baseline / len(channels) if channels else base_spend
                st.session_state[slider_key] = min(balanced_value, max_spend)
            elif scale_up:
                current_val = st.session_state.get(slider_key, base_spend)
                st.session_state[slider_key] = min(current_val * 1.2, max_spend)
            
            if slider_key not in st.session_state:
                st.session_state[slider_key] = base_spend
            
            # Calculate percentage of total budget
            pct_of_total = (base_spend / total_baseline * 100) if total_baseline > 0 else 0
            
            # Channel header with efficiency zone and percentage
            zone = "EFFICIENT" if params.get('marginal_roi', 1.0) > 1.5 else \
                   "DIMINISHING" if params.get('marginal_roi', 1.0) >= 0.8 else "SATURATED"
            
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
            
            current_allocation[ch] = val
            total_simulated += val
            
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
        
        # Total spend summary
        st.markdown("---")
        spend_delta = total_simulated - total_baseline
        spend_pct_delta = ((total_simulated - total_baseline) / total_baseline * 100) if total_baseline > 0 else 0
        delta_display = f"+${spend_delta:,.0f} ({spend_pct_delta:+.1f}%)" if spend_delta >= 0 else f"-${abs(spend_delta):,.0f} ({spend_pct_delta:+.1f}%)"
        
        st.metric(
            "Total Simulated Spend",
            f"${total_simulated:,.0f}",
            delta=delta_display,
            delta_color="normal" if spend_delta >= 0 else "inverse"
        )
        
        # Show baseline total for reference
        st.markdown(
            f"<p style='color: rgba(255,255,255,0.5); font-size: 0.8rem; margin-top: -0.5rem;'>"
            f"Baseline: ${total_baseline:,.0f}</p>",
            unsafe_allow_html=True
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
                mroi = channel_params.get(ch, {}).get('marginal_roi', 1.0)
                zone_at_spend = "EFFICIENT" if mroi > 1.5 else "DIMINISHING" if mroi >= 0.8 else "SATURATED"
            
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
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric(
                "Baseline Revenue",
                f"${baseline_revenue:,.0f}",
                help="Predicted revenue at baseline (historical mean) spend levels"
            )
            # Show that this is fixed/reference value
            st.markdown(
                "<p style='color: rgba(255,255,255,0.4); font-size: 0.7rem; margin-top: -0.5rem;'>"
                "Reference point (fixed)</p>",
                unsafe_allow_html=True
            )
        with col_m2:
            delta_display = f"+${rev_delta:,.0f} ({rev_pct_delta:+.1f}%)" if rev_delta >= 0 else f"-${abs(rev_delta):,.0f} ({rev_pct_delta:+.1f}%)"
            st.metric(
                "Simulated Revenue",
                f"${simulated_revenue:,.0f}",
                delta=delta_display if rev_delta != 0 else "No change",
                delta_color="normal" if rev_delta >= 0 else "inverse"
            )
            # Indicator that this updates with sliders
            st.markdown(
                "<p style='color: rgba(41, 181, 232, 0.7); font-size: 0.7rem; margin-top: -0.5rem;'>"
                "Updates with sliders ↑</p>",
                unsafe_allow_html=True
            )
        with col_m3:
            # CI range
            st.markdown(f"""
            <div style="background: {BG_CARD}; border-radius: 8px; padding: 0.75rem; text-align: center;">
                <div style="color: rgba(255,255,255,0.6); font-size: 0.8rem;">90% Confidence Range</div>
                <div style="color: white; font-weight: 600;">${simulated_ci_lower:,.0f} - ${simulated_ci_upper:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Efficiency Score
        efficiency = calculate_efficiency_score(channel_impacts, df_results)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Portfolio Efficiency Score")
        
        score = efficiency['score']
        score_color = COLOR_SUCCESS if score >= 70 else COLOR_WARNING if score >= 40 else COLOR_DANGER
        
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
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
            max_spend = ch_curve['SPEND'].max()
            fig_curve = add_saturation_zone_annotation(fig_curve, gamma, max_spend)
            
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
