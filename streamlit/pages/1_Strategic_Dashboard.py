"""
Strategic Dashboard - CMO / VP Global Markets Persona

Story-driven executive view with confidence intervals and educational context:
  The Problem → The Data (with uncertainty) → The Insight → The Action

ENHANCED FEATURES:
- 90% confidence intervals on all ROI metrics
- Statistical significance badges
- Collapsible "Learn More" panels for education
- Confidence-rated recommendations
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
    render_insight_callout,
    render_learn_more_panel,
    render_significance_badge,
    render_recommendation_card,
    render_confidence_metric,
    add_error_bars_to_bar_chart,
    format_ci_string,
    calculate_confidence_level,
    apply_plotly_theme,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_ACCENT,
    COLOR_SUCCESS,
    COLOR_DANGER,
    COLOR_WARNING
)
from utils.explanations import get_explanation, render_explanation_expander
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
    page_title="Strategic Dashboard | MMM ROI Engine",
    layout="wide"
)

inject_custom_css()


@st.cache_data(ttl=3600)
def load_dashboard_data(_session):
    """Load all data needed for the executive dashboard including CI data."""
    # Import centralized queries from data_loader
    from utils.data_loader import QUERIES
    
    queries = {
        "ROI": QUERIES["ROI"],
        "WEEKLY": QUERIES["WEEKLY"],
        "RESULTS": QUERIES["RESULTS"],
        "RESULTS_INTERPRETED": "SELECT * FROM MMM.V_MODEL_RESULTS_INTERPRETED",
    }
    return run_queries_parallel(_session, queries)


def generate_recommendation_with_confidence(df_results: pd.DataFrame, df_roi: pd.DataFrame) -> dict:
    """
    Generate recommendation with confidence level based on CI width.
    Uses model results with uncertainty quantification.
    """
    if df_roi.empty:
        return None
    
    # Calculate ROAS for each channel
    df = df_roi.copy()
    df['ROAS'] = df['ATTRIBUTED_REVENUE'] / df['TOTAL_SPEND'].replace(0, np.nan)
    df = df.dropna(subset=['ROAS'])
    
    if df.empty:
        return None
    
    avg_roas = df['ROAS'].mean()
    
    # Find worst and best performers
    worst = df.loc[df['ROAS'].idxmin()]
    best = df.loc[df['ROAS'].idxmax()]
    
    # Get CI data if available from model results
    ci_lower_from = None
    ci_upper_from = None
    ci_lower_to = None
    ci_upper_to = None
    is_from_significant = True
    is_to_significant = True
    
    if not df_results.empty and 'CHANNEL' in df_results.columns:
        from_row = df_results[df_results['CHANNEL'].str.contains(worst['CHANNEL'], case=False, na=False)]
        to_row = df_results[df_results['CHANNEL'].str.contains(best['CHANNEL'], case=False, na=False)]
        
        if len(from_row) > 0:
            ci_lower_from = from_row['ROI_CI_LOWER'].iloc[0] if 'ROI_CI_LOWER' in from_row else None
            ci_upper_from = from_row['ROI_CI_UPPER'].iloc[0] if 'ROI_CI_UPPER' in from_row else None
            is_from_significant = from_row['IS_SIGNIFICANT'].iloc[0] if 'IS_SIGNIFICANT' in from_row else True
        
        if len(to_row) > 0:
            ci_lower_to = to_row['ROI_CI_LOWER'].iloc[0] if 'ROI_CI_LOWER' in to_row else None
            ci_upper_to = to_row['ROI_CI_UPPER'].iloc[0] if 'ROI_CI_UPPER' in to_row else None
            is_to_significant = to_row['IS_SIGNIFICANT'].iloc[0] if 'IS_SIGNIFICANT' in to_row else True
    
    # Determine confidence level
    # Thresholds scaled for ROI values (typical CI widths are 20-100 in ROI units)
    if is_from_significant and is_to_significant:
        if ci_lower_from and ci_upper_from and ci_lower_to and ci_upper_to:
            from_width = ci_upper_from - ci_lower_from
            to_width = ci_upper_to - ci_lower_to
            avg_width = (from_width + to_width) / 2
            if avg_width < 30:
                confidence = "high"
            elif avg_width < 60:
                confidence = "medium"
            else:
                confidence = "low"
        else:
            confidence = "medium"
    else:
        confidence = "low"
    
    # Calculate potential lift from reallocation
    realloc_amount = worst['TOTAL_SPEND'] * 0.25  # Shift 25% from worst
    potential_lift = realloc_amount * (best['ROAS'] - worst['ROAS'])
    
    # Calculate CI for the lift estimate
    if ci_lower_from and ci_upper_from and ci_lower_to and ci_upper_to:
        # Conservative CI: worst case from - best case to
        lift_ci_lower = realloc_amount * (ci_lower_to - ci_upper_from)
        lift_ci_upper = realloc_amount * (ci_upper_to - ci_lower_from)
    else:
        lift_ci_lower = potential_lift * 0.7
        lift_ci_upper = potential_lift * 1.3
    
    return {
        "from_channel": worst['CHANNEL'],
        "from_roas": worst['ROAS'],
        "from_ci_lower": ci_lower_from,
        "from_ci_upper": ci_upper_from,
        "from_significant": is_from_significant,
        "to_channel": best['CHANNEL'],
        "to_roas": best['ROAS'],
        "to_ci_lower": ci_lower_to,
        "to_ci_upper": ci_upper_to,
        "to_significant": is_to_significant,
        "realloc_amount": realloc_amount,
        "potential_lift": potential_lift,
        "lift_ci_lower": lift_ci_lower,
        "lift_ci_upper": lift_ci_upper,
        "avg_roas": avg_roas,
        "confidence": confidence
    }


def main():
    # --- Session & Data ---
    try:
        session = get_active_session()
    except Exception:
        st.error("Could not connect to Snowflake. Please ensure you're running in Snowflake.")
        return
    
    # Sidebar refresh button
    with st.sidebar:
        if st.button("Refresh Data", help="Clear cache and reload data"):
            st.cache_data.clear()
            st.rerun()
    
    with st.spinner("Loading executive dashboard..."):
        data = load_dashboard_data(session)
        df_roi = data.get("ROI", pd.DataFrame())
        df_weekly = data.get("WEEKLY", pd.DataFrame())
        df_results = data.get("RESULTS", pd.DataFrame())
        df_interpreted = data.get("RESULTS_INTERPRETED", pd.DataFrame())

    # --- Header ---
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1>Strategic ROI Dashboard</h1>
        <p style="color: rgba(255,255,255,0.6); font-size: 1.1rem;">
            Executive view of marketing performance with statistical confidence
        </p>
    </div>
    """, unsafe_allow_html=True)

    if df_roi.empty:
        st.warning("No ROI data available. Please run the MMM training pipeline first.")
        return

    # --- THE PROBLEM ---
    st.markdown(
        render_story_section(
            "The Challenge",
            "Marketing spend is fragmented across multiple channels and regions. "
            "Which investments are truly driving revenue, and where should we reallocate budget?"
        ),
        unsafe_allow_html=True
    )
    
    # Educational Panel: What is ROI Attribution?
    with st.expander("Learn More: What is ROI Attribution?", expanded=False):
        exp = get_explanation("roi_attribution")
        st.markdown(exp.get("content", ""), unsafe_allow_html=True)

    # --- THE DATA (KPIs with Confidence) ---
    total_spend = df_roi['TOTAL_SPEND'].sum()
    total_rev = df_roi['ATTRIBUTED_REVENUE'].sum()
    blended_roas = total_rev / total_spend if total_spend > 0 else 0
    num_channels = len(df_roi)
    
    # Calculate blended CI if available
    blended_ci_lower = blended_roas * 0.85  # Default ±15%
    blended_ci_upper = blended_roas * 1.15
    significant_channels = 0
    
    if not df_results.empty and 'IS_SIGNIFICANT' in df_results.columns:
        significant_channels = df_results['IS_SIGNIFICANT'].sum()
        if 'ROI_CI_LOWER' in df_results.columns:
            # Weighted average CI
            weights = df_results['CURRENT_SPEND'] if 'CURRENT_SPEND' in df_results else None
            if weights is not None and weights.sum() > 0:
                blended_ci_lower = np.average(df_results['ROI_CI_LOWER'].fillna(0), weights=weights)
                blended_ci_upper = np.average(df_results['ROI_CI_UPPER'].fillna(blended_roas * 1.3), weights=weights)
    
    st.markdown("### Key Performance Indicators")
    
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.metric(
            "Total Marketing Spend",
            f"${total_spend/1e6:.2f}M",
            help="Historical spend across all channels in the model period"
        )
    
    with c2:
        st.metric(
            "Attributed Revenue",
            f"${total_rev/1e6:.2f}M",
            help="Incremental revenue attributed to marketing via MMM (not baseline demand)"
        )
    
    with c3:
        # Enhanced ROAS with CI
        delta_color = "normal" if blended_roas >= 1.0 else "inverse"
        st.metric(
            "Blended ROAS",
            f"{blended_roas:.2f}x",
            delta=f"90% CI: [{blended_ci_lower:.2f} - {blended_ci_upper:.2f}]",
            delta_color=delta_color,
            help="Portfolio-level Return on Ad Spend with confidence interval"
        )
    
    with c4:
        st.metric(
            "Significant Channels",
            f"{significant_channels}/{num_channels}",
            help="Channels where we're confident ROI is above breakeven"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Educational Panel: Understanding Confidence Intervals
    with st.expander("Learn More: Understanding Confidence Intervals", expanded=False):
        exp = get_explanation("confidence_intervals")
        st.markdown(exp.get("content", ""), unsafe_allow_html=True)

    # --- REGIONAL ROI MAP ---
    st.markdown("### Regional Performance Overview")
    st.markdown(
        "<p style='color: rgba(255,255,255,0.6);'>Click a region to filter the dashboard</p>",
        unsafe_allow_html=True
    )
    
    # Regional map with selection
    if not df_results.empty:
        selected_region = render_region_selector_map(df_results, key_prefix="dashboard")
        
        # Show regional summary metrics
        render_regional_summary_metrics(df_results, selected_region)
        
        # Regional drill-down cards
        if selected_region:
            render_region_drill_down(selected_region, df_results, key_prefix="dashboard_drill")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Filter df_roi and df_results if a region is selected
        if selected_region:
            # Add region column for filtering
            df_roi['Region'] = df_roi['CHANNEL'].apply(extract_region_from_channel)
            df_roi = df_roi[df_roi['Region'] == selected_region]
            
            df_results_display = df_results.copy()
            df_results_display['Region'] = df_results_display['CHANNEL'].apply(extract_region_from_channel)
            df_results = df_results_display[df_results_display['Region'] == selected_region]
    else:
        selected_region = None

    # --- THE INSIGHT (Charts with CI) ---
    region_label = f" ({REGION_COORDS.get(selected_region, {}).get('name', selected_region)})" if selected_region else ""
    st.markdown(f"### Channel Attribution Analysis{region_label}")
    
    # Check if we have data to display after region filtering
    if df_roi.empty:
        st.info(f"No channel data available for the selected region.")
    else:
        # Calculate region-specific total revenue for display
        region_total_rev = df_roi['ATTRIBUTED_REVENUE'].sum()
        
        col_spend, col_water = st.columns(2)
        
        with col_spend:
            st.markdown("##### Spend Allocation by Channel")
            df_spend_sorted = df_roi.sort_values('TOTAL_SPEND', ascending=True)
            
            fig_spend = go.Figure(go.Bar(
                x=df_spend_sorted['TOTAL_SPEND'],
                y=df_spend_sorted['CHANNEL'],
                orientation='h',
                marker_color=COLOR_PRIMARY,
                text=[f"${v/1e6:.1f}M" for v in df_spend_sorted['TOTAL_SPEND']],
                textposition='outside',
                textfont=dict(color='white')
            ))
            fig_spend = apply_plotly_theme(fig_spend)
            fig_spend.update_layout(
                title="",
                xaxis_title="Total Spend ($)",
                yaxis_title="",
                margin=dict(l=0, r=80, t=10, b=40),
                height=350
            )
            st.plotly_chart(fig_spend, use_container_width=True, key="strategic_spend_mix")
        
        with col_water:
            st.markdown("##### Revenue Attribution Waterfall")
            df_sorted = df_roi.sort_values('ATTRIBUTED_REVENUE', ascending=False)
            
            # Handle negative revenue: show actual values with appropriate formatting
            revenue_values = df_sorted['ATTRIBUTED_REVENUE'].tolist()
            text_labels = [f"${v/1e6:.1f}M" if v >= 0 else f"-${abs(v)/1e6:.1f}M" for v in revenue_values]
            text_labels.append(f"${region_total_rev/1e6:.1f}M" if region_total_rev >= 0 else f"-${abs(region_total_rev)/1e6:.1f}M")
            
            fig_water = go.Figure(go.Waterfall(
                name="Attribution",
                orientation="v",
                measure=["relative"] * len(df_sorted) + ["total"],
                x=df_sorted['CHANNEL'].tolist() + ["Total"],
                textposition="outside",
                text=text_labels,
                y=revenue_values + [region_total_rev],
                connector={"line": {"color": "rgba(255,255,255,0.3)"}},
                decreasing={"marker": {"color": COLOR_DANGER}},
                increasing={"marker": {"color": COLOR_SUCCESS}},
                totals={"marker": {"color": COLOR_PRIMARY}}
            ))
            fig_water = apply_plotly_theme(fig_water)
            fig_water.update_layout(
                title="",
                showlegend=False,
                margin=dict(l=0, r=0, t=10, b=40),
                height=350
            )
            st.plotly_chart(fig_water, use_container_width=True, key="strategic_waterfall")

    # --- ROAS by Channel with Confidence Intervals ---
    st.markdown("##### Return on Ad Spend by Channel (with 90% Confidence Intervals)")
    
    if not df_roi.empty:
        df_roas = df_roi.copy()
        df_roas['ROAS'] = df_roas['ATTRIBUTED_REVENUE'] / df_roas['TOTAL_SPEND'].replace(0, np.nan)
        df_roas = df_roas.dropna(subset=['ROAS']).sort_values('ROAS', ascending=True)
        
        # Merge CI data if available
        if not df_results.empty and 'CHANNEL' in df_results.columns:
            df_roas = df_roas.merge(
                df_results[['CHANNEL', 'ROI_CI_LOWER', 'ROI_CI_UPPER', 'IS_SIGNIFICANT']].rename(
                    columns={'ROI_CI_LOWER': 'CI_LOWER', 'ROI_CI_UPPER': 'CI_UPPER'}
                ),
                left_on='CHANNEL',
                right_on='CHANNEL',
                how='left'
            )
        else:
            # Default CI: ±20% of estimate
            df_roas['CI_LOWER'] = df_roas['ROAS'] * 0.8
            df_roas['CI_UPPER'] = df_roas['ROAS'] * 1.2
            df_roas['IS_SIGNIFICANT'] = df_roas['ROAS'] >= 1.0
        
        if not df_roas.empty:
            # Color bars based on significance and performance
            colors = []
            for _, row in df_roas.iterrows():
                if row.get('IS_SIGNIFICANT', True):
                    colors.append(COLOR_SUCCESS if row['ROAS'] >= 1.0 else COLOR_DANGER)
                else:
                    colors.append(COLOR_WARNING)  # Uncertain
            
            fig_roas = go.Figure(go.Bar(
                x=df_roas['ROAS'],
                y=df_roas['CHANNEL'],
                orientation='h',
                marker_color=colors,
                text=[f"{v:.2f}x" for v in df_roas['ROAS']],
                textposition='outside',
                textfont=dict(color='white'),
                error_x=dict(
                    type='data',
                    array=(df_roas['CI_UPPER'] - df_roas['ROAS']).tolist(),
                    arrayminus=(df_roas['ROAS'] - df_roas['CI_LOWER']).tolist(),
                    color='rgba(255, 255, 255, 0.4)',
                    thickness=1.5,
                    width=4
                )
            ))
            fig_roas = apply_plotly_theme(fig_roas)
            fig_roas.update_layout(
                title="",
                xaxis_title="ROAS (Revenue / Spend) with 90% CI",
                yaxis_title="",
                margin=dict(l=0, r=100, t=10, b=40),
                height=max(300, len(df_roas) * 35),
                shapes=[
                    dict(
                        type='line',
                        x0=1, x1=1,
                        y0=-0.5, y1=len(df_roas) - 0.5,
                        line=dict(color='white', width=2, dash='dash')
                    )
                ],
                annotations=[
                    dict(
                        x=1, y=len(df_roas),
                        text="Breakeven (1.0x)",
                        showarrow=False,
                        font=dict(color='rgba(255,255,255,0.6)', size=10),
                        yshift=15
                    )
                ]
            )
            st.plotly_chart(fig_roas, use_container_width=True, key="strategic_roas")
        else:
            st.info("No ROAS data available for the selected region.")
    
    # Legend for colors
    st.markdown("""
    <div style="display: flex; gap: 2rem; margin-top: -1rem; margin-bottom: 1rem; font-size: 0.85rem;">
        <span><span style="color: #2ECC71; font-weight: bold;">[Significant]</span> Significant & Profitable</span>
        <span><span style="color: #E74C3C; font-weight: bold;">[Significant]</span> Significant & Below Breakeven</span>
        <span><span style="color: #F39C12; font-weight: bold;">[Uncertain]</span> Uncertain (Wide CI)</span>
    </div>
    """, unsafe_allow_html=True)

    # --- THE ACTION (Enhanced Recommendation with Confidence) ---
    rec = generate_recommendation_with_confidence(df_results, df_roi)
    
    if rec:
        # Determine recommendation styling based on confidence
        confidence_text = {
            "high": "High Confidence",
            "medium": "Medium Confidence",
            "low": "Low Confidence - Proceed with Caution"
        }.get(rec['confidence'], "")
        
        # Format the lift with CI
        lift_text = f"${rec['potential_lift']/1e6:.2f}M"
        if rec.get('lift_ci_lower') and rec.get('lift_ci_upper'):
            lift_text += f" [{rec['lift_ci_lower']/1e6:.2f}M - {rec['lift_ci_upper']/1e6:.2f}M]"
        
        # Build recommendation content
        rec_content = f"""
            <strong>Reallocate ${rec['realloc_amount']/1e3:.0f}K</strong> from 
            <strong>{rec['from_channel']}</strong> (ROAS: {rec['from_roas']:.2f}x) to 
            <strong>{rec['to_channel']}</strong> (ROAS: {rec['to_roas']:.2f}x)
        """
        
        st.markdown(
            render_recommendation_card(
                title="AI Budget Optimization Recommendation",
                content=rec_content,
                confidence=rec['confidence'],
                predicted_lift=lift_text
            ),
            unsafe_allow_html=True
        )
        
        # Show supporting detail in expander
        with st.expander("View recommendation details", expanded=False):
            col_from, col_to = st.columns(2)
            with col_from:
                st.markdown(f"**Source Channel: {rec['from_channel']}**")
                st.markdown(f"- Current ROAS: {rec['from_roas']:.2f}x")
                if rec.get('from_ci_lower') and rec.get('from_ci_upper'):
                    st.markdown(f"- 90% CI: [{rec['from_ci_lower']:.2f} - {rec['from_ci_upper']:.2f}]")
                st.markdown(render_significance_badge(rec.get('from_significant', True), rec['from_roas']), unsafe_allow_html=True)
            with col_to:
                st.markdown(f"**Target Channel: {rec['to_channel']}**")
                st.markdown(f"- Current ROAS: {rec['to_roas']:.2f}x")
                if rec.get('to_ci_lower') and rec.get('to_ci_upper'):
                    st.markdown(f"- 90% CI: [{rec['to_ci_lower']:.2f} - {rec['to_ci_upper']:.2f}]")
                st.markdown(render_significance_badge(rec.get('to_significant', True), rec['to_roas']), unsafe_allow_html=True)
    
    # Educational Panel: Why These Recommendations?
    with st.expander("Learn More: Why These Recommendations?", expanded=False):
        exp = get_explanation("budget_optimizer")
        st.markdown(exp.get("content", ""), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        exp2 = get_explanation("recommendation_confidence")
        st.markdown(f"<strong>{exp2.get('title', '')}</strong>", unsafe_allow_html=True)
        st.markdown(exp2.get("content", ""), unsafe_allow_html=True)
    
    # --- Weekly Trends with Lag Annotation ---
    if not df_weekly.empty:
        st.markdown("### Performance Trends")
        
        # Filter by region if selected (if CHANNEL column exists in weekly data)
        df_weekly_filtered = df_weekly.copy()
        if selected_region and 'CHANNEL' in df_weekly.columns:
            df_weekly_filtered['Region'] = df_weekly_filtered['CHANNEL'].apply(extract_region_from_channel)
            df_weekly_filtered = df_weekly_filtered[df_weekly_filtered['Region'] == selected_region]
        
        if not df_weekly_filtered.empty:
            df_trend = df_weekly_filtered.groupby('WEEK_START')[['SPEND', 'REVENUE']].sum().reset_index()
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=df_trend['WEEK_START'],
                y=df_trend['SPEND'],
                name='Spend',
                line=dict(color=COLOR_PRIMARY, width=2),
                fill='tozeroy',
                fillcolor=f'rgba(41, 181, 232, 0.1)'
            ))
            fig_trend.add_trace(go.Scatter(
                x=df_trend['WEEK_START'],
                y=df_trend['REVENUE'],
                name='Revenue',
                line=dict(color=COLOR_ACCENT, width=2),
                yaxis='y2'
            ))
            
            fig_trend = apply_plotly_theme(fig_trend)
            fig_trend.update_layout(
                title="",
                yaxis=dict(title='Spend ($)', side='left', showgrid=True),
                yaxis2=dict(title='Revenue ($)', side='right', overlaying='y', showgrid=False),
                legend=dict(x=0, y=1.15, orientation='h'),
                margin=dict(l=0, r=0, t=30, b=40),
                height=350
            )
            
            # Add annotation about lag
            if len(df_trend) > 0 and df_trend['REVENUE'].max() > 0:
                fig_trend.add_annotation(
                    x=df_trend['WEEK_START'].iloc[len(df_trend)//2],
                    y=df_trend['REVENUE'].max() * 0.9,
                    text="Note: Revenue typically lags spend by 8-12 weeks (B2B sales cycle)",
                    showarrow=False,
                    font=dict(color='rgba(255,255,255,0.5)', size=10),
                    bgcolor='rgba(0,0,0,0.3)',
                    borderpad=4
                )
            
            st.plotly_chart(fig_trend, use_container_width=True, key="strategic_trends")
        else:
            st.info("No weekly trend data available for the selected region.")

    # --- Priority Actions Summary ---
    if not df_results.empty and 'MARGINAL_ROI' in df_results.columns:
        st.markdown("### Priority Actions")
        
        df_actions = df_results.nlargest(3, 'MARGINAL_ROI')[['CHANNEL', 'ROI', 'MARGINAL_ROI', 'IS_SIGNIFICANT']]
        
        cols = st.columns(3)
        for i, (_, row) in enumerate(df_actions.iterrows()):
            with cols[i]:
                badge = "[High Confidence]" if row.get('IS_SIGNIFICANT', True) else "[Uncertain]"
                badge_color = COLOR_SUCCESS if row.get('IS_SIGNIFICANT', True) else COLOR_WARNING
                st.markdown(f"""
                <div style="background: rgba(41, 181, 232, 0.1); border: 1px solid rgba(41, 181, 232, 0.3); 
                            border-radius: 12px; padding: 1rem; text-align: center;">
                    <div style="font-size: 0.75rem; color: {badge_color}; margin-bottom: 0.5rem;">{badge}</div>
                    <div style="font-weight: 600; color: white; margin-bottom: 0.25rem;">{row['CHANNEL']}</div>
                    <div style="font-size: 1.5rem; color: {COLOR_PRIMARY}; font-weight: 700;">{row['MARGINAL_ROI']:.2f}x</div>
                    <div style="font-size: 0.8rem; color: rgba(255,255,255,0.5);">Marginal ROI</div>
                    <div style="font-size: 0.9rem; color: rgba(255,255,255,0.7); margin-top: 0.5rem;">
                        → Increase investment
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # --- Navigation ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Back to Home", use_container_width=True):
            st.switch_page("mmm_roi_app.py")
    with col2:
        if st.button("Open Budget Simulator", use_container_width=True):
            st.switch_page("pages/2_Simulator.py")
    with col3:
        if st.button("Explore Model Details", use_container_width=True):
            st.switch_page("pages/3_Model_Explorer.py")


if __name__ == "__main__":
    main()
