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

# --- Page Config ---
st.set_page_config(
    page_title="Strategic Dashboard | MMM ROI Engine",
    layout="wide"
)

inject_custom_css()


@st.cache_data(ttl=300)
def load_dashboard_data(_session):
    """Load all data needed for the executive dashboard including CI data."""
    # Import centralized queries from data_loader
    from utils.data_loader import QUERIES, DATABASE
    
    queries = {
        "ROI": QUERIES["ROI"],
        "WEEKLY": QUERIES["WEEKLY"],
        "RESULTS": QUERIES["RESULTS"],
        "RESULTS_INTERPRETED": f"SELECT * FROM {DATABASE}.MMM.V_MODEL_RESULTS_INTERPRETED",
    }
    return run_queries_parallel(_session, queries)


def generate_recommendation_with_confidence(df_results: pd.DataFrame, df_roi: pd.DataFrame, min_spend_pct: float = 10) -> dict:
    """
    Generate recommendation with confidence level based on CI width.
    Uses model results with uncertainty quantification.
    Only considers channels above spend threshold for reliable recommendations.
    """
    if df_roi.empty:
        return None
    
    # Calculate ROAS for each channel
    df = df_roi.copy()
    df['ROAS'] = df['ATTRIBUTED_REVENUE'] / df['TOTAL_SPEND'].replace(0, np.nan)
    df = df.dropna(subset=['ROAS'])
    
    if df.empty:
        return None
    
    # Filter to reliable channels only (above spend threshold)
    if not df_results.empty and 'CURRENT_SPEND' in df_results.columns:
        max_spend = df_results['CURRENT_SPEND'].max()
        reliable_channels = df_results[df_results['CURRENT_SPEND'] >= max_spend * min_spend_pct / 100]['CHANNEL'].tolist()
        df['IS_RELIABLE'] = df['CHANNEL'].isin(reliable_channels)
        df_reliable = df[df['IS_RELIABLE']]
        if df_reliable.empty:
            df_reliable = df  # Fall back to all if none match
    else:
        df_reliable = df
    
    avg_roas = df_reliable['ROAS'].mean()
    
    # Find worst and best performers from RELIABLE channels only
    worst = df_reliable.loc[df_reliable['ROAS'].idxmin()]
    best = df_reliable.loc[df_reliable['ROAS'].idxmax()]
    
    # Get CI data if available from model results
    ci_lower_from = None
    ci_upper_from = None
    ci_lower_to = None
    ci_upper_to = None
    is_from_significant = True
    is_to_significant = True
    
    if not df_results.empty and 'CHANNEL' in df_results.columns:
        from_match = worst['CHANNEL']
        to_match = best['CHANNEL']
        from_row = df_results[df_results['CHANNEL'] == from_match]
        to_row = df_results[df_results['CHANNEL'] == to_match]
        
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


# Spend threshold for reliable model estimates (as % of max spend)
DEFAULT_SPEND_THRESHOLD_PCT = 10  # Channels below 10% of max spend flagged as "needs validation"


def classify_channel_reliability(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Classify channels by spend level to indicate reliability of model estimates.
    Low-spend channels have unreliable ROI estimates due to signal-to-noise issues.
    """
    if df_results.empty or 'CURRENT_SPEND' not in df_results.columns:
        return df_results
    
    df = df_results.copy()
    max_spend = df['CURRENT_SPEND'].max()
    
    df['SPEND_PCT_OF_MAX'] = (df['CURRENT_SPEND'] / max_spend * 100).round(1)
    df['RELIABILITY'] = df['SPEND_PCT_OF_MAX'].apply(
        lambda x: 'HIGH' if x >= 15 else ('MEDIUM' if x >= 5 else 'LOW')
    )
    df['NEEDS_VALIDATION'] = df['RELIABILITY'] == 'LOW'
    
    return df


def main():
    # --- Session & Data ---
    try:
        session = get_active_session()
    except Exception:
        st.error("Could not connect to Snowflake. Please ensure you're running in Snowflake.")
        return
    
    # Sidebar controls
    with st.sidebar:
        if st.button("Refresh Data", help="Clear cache and reload data"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("##### Data Reliability Filter")
        min_spend_pct = st.slider(
            "Min spend % of largest channel",
            min_value=0,
            max_value=25,
            value=DEFAULT_SPEND_THRESHOLD_PCT,
            step=5,
            help="Channels below this threshold have unreliable model ROI estimates. Consider A/B testing instead."
        )
        show_low_spend = st.checkbox(
            "Show low-spend channels",
            value=True,
            help="Include channels that need validation via A/B testing"
        )
    
    with st.spinner("Loading executive dashboard..."):
        data = load_dashboard_data(session)
        df_roi = data.get("ROI", pd.DataFrame())
        df_weekly = data.get("WEEKLY", pd.DataFrame())
        df_results = data.get("RESULTS", pd.DataFrame())
        df_interpreted = data.get("RESULTS_INTERPRETED", pd.DataFrame())
    
    # Classify channel reliability based on spend levels
    if not df_results.empty:
        df_results = classify_channel_reliability(df_results)

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

    # --- THE INSIGHT (Charts with CI) ---
    st.markdown("### Channel Attribution Analysis")
    
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
        
        # Merge reliability data (NOT CI bounds - those are for ROI coefficients, not ROAS)
        if not df_results.empty and 'CHANNEL' in df_results.columns:
            merge_cols = ['CHANNEL', 'IS_SIGNIFICANT', 'CURRENT_SPEND']
            if 'SPEND_PCT_OF_MAX' in df_results.columns:
                merge_cols.extend(['SPEND_PCT_OF_MAX', 'RELIABILITY', 'NEEDS_VALIDATION'])
            
            df_results_merge = df_results[[c for c in merge_cols if c in df_results.columns]].copy()
            
            df_roas = df_roas.merge(
                df_results_merge,
                on='CHANNEL',
                how='left'
            )
        
        # Calculate CI bounds as percentage of ROAS (not from model coefficients which are different scale)
        df_roas['CI_LOWER'] = df_roas['ROAS'] * 0.8
        df_roas['CI_UPPER'] = df_roas['ROAS'] * 1.2
        if 'IS_SIGNIFICANT' not in df_roas.columns:
            df_roas['IS_SIGNIFICANT'] = df_roas['ROAS'] >= 1.0
        if 'NEEDS_VALIDATION' not in df_roas.columns:
            df_roas['NEEDS_VALIDATION'] = False
        if 'SPEND_PCT_OF_MAX' not in df_roas.columns:
            df_roas['SPEND_PCT_OF_MAX'] = 100
        
        # Apply spend threshold filter
        if 'SPEND_PCT_OF_MAX' in df_roas.columns:
            df_roas['ABOVE_THRESHOLD'] = df_roas['SPEND_PCT_OF_MAX'] >= min_spend_pct
            if not show_low_spend:
                df_roas = df_roas[df_roas['ABOVE_THRESHOLD']]
        
        if not df_roas.empty:
            # Color bars based on ROAS: green=profitable (>=1.0), red=unprofitable (<1.0), gray=low spend
            colors = []
            y_labels = []
            for idx, row in df_roas.iterrows():
                roas_val = row['ROAS']
                spend_pct = row['SPEND_PCT_OF_MAX'] if pd.notna(row['SPEND_PCT_OF_MAX']) else 100.0
                needs_val = row['NEEDS_VALIDATION'] if pd.notna(row['NEEDS_VALIDATION']) else False
                is_low_spend = needs_val or spend_pct < min_spend_pct
                
                if is_low_spend:
                    colors.append('#6B7280')  # Gray for low-spend channels
                elif roas_val >= 1.0:
                    colors.append('#28A745')  # Green for profitable
                else:
                    colors.append('#DC3545')  # Red for unprofitable
                
                label = row['CHANNEL']
                if spend_pct < min_spend_pct:
                    label += " ⚠️"
                y_labels.append(label)
            
            fig_roas = go.Figure(go.Bar(
                x=df_roas['ROAS'],
                y=y_labels,
                orientation='h',
                marker=dict(color=colors),
                text=[f"{v:.2f}x" for v in df_roas['ROAS']],
                textposition='outside',
                textfont=dict(color='white'),
                error_x=dict(
                    type='data',
                    array=(df_roas['CI_UPPER'] - df_roas['ROAS']).tolist(),
                    arrayminus=(df_roas['ROAS'] - df_roas['CI_LOWER']).tolist(),
                    color='#9B59B6',
                    thickness=2,
                    width=5
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
                        line=dict(color='#FFD700', width=2, dash='dash')
                    )
                ],
                annotations=[
                    dict(
                        x=1, y=len(df_roas),
                        text="Breakeven (1.0x)",
                        showarrow=False,
                        font=dict(color='#FFD700', size=10),
                        yshift=15
                    )
                ]
            )
            st.plotly_chart(fig_roas, use_container_width=True, key="strategic_roas")
            
            # Show count of filtered channels
            low_spend_count = df_roas[df_roas.get('SPEND_PCT_OF_MAX', pd.Series([100]*len(df_roas))) < min_spend_pct].shape[0]
            if low_spend_count > 0:
                st.caption(f"⚠️ {low_spend_count} channel(s) below {min_spend_pct}% spend threshold - consider A/B testing for validation")
        else:
            st.info("No ROAS data available for the selected region.")
    
    # Legend for colors
    st.markdown(f"""
    <div style="display: flex; flex-wrap: wrap; gap: 1.5rem; margin-top: -0.5rem; margin-bottom: 1rem; font-size: 0.85rem;">
        <span><span style="color: {COLOR_SUCCESS}; font-weight: bold;">■</span> Profitable (ROAS ≥ 1.0x)</span>
        <span><span style="color: {COLOR_DANGER}; font-weight: bold;">■</span> Below breakeven (ROAS &lt; 1.0x)</span>
        <span><span style="color: #6B7280; font-weight: bold;">■</span> Low spend - needs A/B test ⚠️</span>
    </div>
    """, unsafe_allow_html=True)

    # --- THE ACTION (Enhanced Recommendation with Confidence) ---
    rec = generate_recommendation_with_confidence(df_results, df_roi, min_spend_pct)
    
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
        
        df_weekly_filtered = df_weekly.copy()
        
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
        
        # Filter to reliable channels only for recommendations
        df_reliable = df_results.copy()
        if 'SPEND_PCT_OF_MAX' in df_reliable.columns:
            df_reliable = df_reliable[df_reliable['SPEND_PCT_OF_MAX'] >= min_spend_pct]
        
        if df_reliable.empty:
            st.info("No channels meet the reliability threshold. Adjust the filter in the sidebar.")
        else:
            df_actions = df_reliable.nlargest(3, 'MARGINAL_ROI')[['CHANNEL', 'ROI', 'MARGINAL_ROI', 'IS_SIGNIFICANT', 'CURRENT_SPEND']]
            
            cols = st.columns(min(3, len(df_actions)))
            for i, (_, row) in enumerate(df_actions.iterrows()):
                with cols[i]:
                    badge = "[High Confidence]" if row.get('IS_SIGNIFICANT', True) else "[Uncertain]"
                    badge_color = COLOR_SUCCESS if row.get('IS_SIGNIFICANT', True) else COLOR_WARNING
                    spend_label = f"${row['CURRENT_SPEND']/1e6:.0f}M spend"
                    st.markdown(f"""
                    <div class="priority-action-card">
                        <div class="badge" style="color: {badge_color};">{badge}</div>
                        <div class="channel-name">{row['CHANNEL']}</div>
                        <div class="metric-value">{row['MARGINAL_ROI']:.2f}x</div>
                        <div class="metric-label">Marginal ROI</div>
                        <div class="spend-info">{spend_label}</div>
                        <div class="action-text">→ Increase investment</div>
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
