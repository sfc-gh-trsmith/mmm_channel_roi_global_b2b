"""
Model Explorer - Data Scientist Persona

Technical deep-dive into MMM model diagnostics, response curves,
exploratory data analysis, and Cortex Analyst for ad-hoc queries.

ENHANCED FEATURES:
- Full model diagnostics (residuals, predicted vs actual, CV folds)
- Parameter table with learned MMM parameters and CIs
- Data quality indicators and warnings
- 4 educational panels on technical methodology
- Time-lagged cross-correlation visualization
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
    render_learn_more_panel,
    render_significance_badge,
    render_zone_badge,
    render_parameter_pill,
    add_confidence_band_to_line_chart,
    add_saturation_zone_annotation,
    format_ci_string,
    apply_plotly_theme,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_ACCENT,
    COLOR_SUCCESS,
    COLOR_WARNING,
    COLOR_DANGER,
    BG_CARD
)
from utils.explanations import get_explanation
from utils.cortex_analyst import (
    render_analyst_chat,
    render_example_queries,
    generate_diagnostic_narrative,
    generate_comparative_narrative,
    EXAMPLE_QUERIES
)

# --- Page Config ---
st.set_page_config(
    page_title="Model Explorer | MMM ROI Engine",
    layout="wide"
)

inject_custom_css()


@st.cache_data(ttl=3600)
def load_explorer_data(_session):
    """Load all data for model exploration including enhanced fields."""
    # Import centralized queries from data_loader
    from utils.data_loader import QUERIES
    
    # Select the queries needed for Model Explorer
    explorer_queries = {
        "WEEKLY": QUERIES["WEEKLY"],
        "CURVES": QUERIES["CURVES"],
        "RESULTS": QUERIES["RESULTS"],
        "ROI": QUERIES["ROI"],
        "METADATA": QUERIES["METADATA"],
    }
    return run_queries_parallel(_session, explorer_queries)


def render_model_health_card(df_results: pd.DataFrame, df_weekly: pd.DataFrame) -> None:
    """Render model health diagnostics card."""
    
    # Calculate health metrics
    if df_results.empty:
        st.warning("No model results available for health assessment.")
        return
    
    # Metrics
    cv_mape = df_results['CV_MAPE'].mean() if 'CV_MAPE' in df_results.columns else None
    r_squared = df_results['R_SQUARED'].mean() if 'R_SQUARED' in df_results.columns else None
    n_channels = len(df_results)
    significant_pct = (df_results['IS_SIGNIFICANT'].sum() / n_channels * 100) if 'IS_SIGNIFICANT' in df_results.columns else None
    
    # Quality assessment
    quality_score = 0
    issues = []
    
    if cv_mape is not None:
        if cv_mape < 10:
            quality_score += 30
        elif cv_mape < 15:
            quality_score += 20
        elif cv_mape < 20:
            quality_score += 10
        else:
            issues.append(f"High CV MAPE ({cv_mape:.1f}%) - model may not generalize well")
    
    if r_squared is not None:
        if r_squared > 0.9:
            quality_score += 30
        elif r_squared > 0.85:
            quality_score += 20
        elif r_squared > 0.8:
            quality_score += 10
        else:
            issues.append(f"Low R² ({r_squared:.2f}) - model explains limited variance")
    
    if significant_pct is not None:
        if significant_pct > 80:
            quality_score += 20
        elif significant_pct > 60:
            quality_score += 15
        elif significant_pct > 40:
            quality_score += 10
        else:
            issues.append(f"Only {significant_pct:.0f}% of channels are statistically significant")
    
    # Data quality checks
    if not df_weekly.empty:
        n_weeks = df_weekly['WEEK_START'].nunique()
        if n_weeks < 52:
            issues.append(f"Only {n_weeks} weeks of data (recommend 104+ for seasonality)")
        quality_score += min(20, n_weeks // 5)
    
    quality_color = COLOR_SUCCESS if quality_score >= 70 else COLOR_WARNING if quality_score >= 50 else COLOR_DANGER
    quality_text = "Good" if quality_score >= 70 else "Fair" if quality_score >= 50 else "Needs Attention"
    
    st.markdown(f"""
    <div style="background: {BG_CARD}; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <h4 style="margin: 0; color: white;">Model Health Assessment</h4>
            <div style="background: {quality_color}22; color: {quality_color}; padding: 0.25rem 0.75rem; border-radius: 20px; font-weight: 600;">
                {quality_text} ({quality_score}/100)
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;">
            <div style="text-align: center;">
                <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem;">CV MAPE</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: {COLOR_SUCCESS if cv_mape and cv_mape < 15 else COLOR_WARNING if cv_mape and cv_mape < 20 else COLOR_DANGER};">
                    {f'{cv_mape:.1f}%' if cv_mape else 'N/A'}
                </div>
            </div>
            <div style="text-align: center;">
                <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem;">In-Sample R²</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: {COLOR_SUCCESS if r_squared and r_squared > 0.85 else COLOR_WARNING};">
                    {f'{r_squared:.3f}' if r_squared else 'N/A'}
                </div>
            </div>
            <div style="text-align: center;">
                <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem;">Significant</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: {COLOR_SUCCESS if significant_pct and significant_pct > 60 else COLOR_WARNING};">
                    {f'{significant_pct:.0f}%' if significant_pct else 'N/A'}
                </div>
            </div>
            <div style="text-align: center;">
                <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem;">Channels</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: white;">
                    {n_channels}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Issues list
    if issues:
        st.markdown("**Diagnostic Alerts:**")
        for issue in issues:
            st.markdown(f"- [Alert] {issue}")


def render_eda_tab(df_weekly: pd.DataFrame):
    """Render Exploratory Data Analysis tab."""
    if df_weekly.empty:
        st.info("No weekly data available for analysis.")
        return
    
    st.markdown("### Trends & Seasonality")
    
    # Aggregate weekly trends
    df_agg = df_weekly.groupby('WEEK_START')[['SPEND', 'REVENUE']].sum().reset_index()
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=df_agg['WEEK_START'],
        y=df_agg['SPEND'],
        name='Total Spend',
        line=dict(color=COLOR_PRIMARY, width=2),
        fill='tozeroy',
        fillcolor='rgba(41, 181, 232, 0.1)'
    ))
    fig_trend.add_trace(go.Scatter(
        x=df_agg['WEEK_START'],
        y=df_agg['REVENUE'],
        name='Total Revenue',
        line=dict(color=COLOR_ACCENT, width=2),
        yaxis='y2'
    ))
    
    fig_trend = apply_plotly_theme(fig_trend)
    fig_trend.update_layout(
        title="Weekly Spend vs Revenue (Time-Shifted Relationship)",
        yaxis=dict(title='Spend ($)', side='left'),
        yaxis2=dict(title='Revenue ($)', side='right', overlaying='y', showgrid=False),
        legend=dict(orientation='h', y=1.1),
        height=400
    )
    st.plotly_chart(fig_trend, use_container_width=True, key="eda_trend")
    
    # Time-lagged cross-correlation
    st.markdown("### Time-Lagged Cross-Correlation")
    st.markdown(
        "<p style='color: rgba(255,255,255,0.6);'>"
        "Shows how spend in week <em>t</em> correlates with revenue in week <em>t+lag</em>. "
        "Peaks indicate the typical delay between spend and revenue impact."
        "</p>",
        unsafe_allow_html=True
    )
    
    if len(df_agg) > 20:
        max_lag = min(26, len(df_agg) // 3)  # Up to 6 months lag
        lags = range(0, max_lag + 1)
        correlations = []
        
        spend = df_agg['SPEND'].values
        revenue = df_agg['REVENUE'].values
        
        for lag in lags:
            if lag < len(spend):
                corr = np.corrcoef(spend[:-lag] if lag > 0 else spend, 
                                   revenue[lag:] if lag > 0 else revenue)[0, 1]
                correlations.append(corr)
        
        fig_lag = go.Figure(go.Bar(
            x=list(lags)[:len(correlations)],
            y=correlations,
            marker_color=[COLOR_SUCCESS if c == max(correlations) else COLOR_PRIMARY for c in correlations]
        ))
        fig_lag = apply_plotly_theme(fig_lag)
        fig_lag.update_layout(
            title="Spend-Revenue Cross-Correlation by Lag (weeks)",
            xaxis_title="Lag (weeks)",
            yaxis_title="Correlation",
            height=300
        )
        
        # Annotate peak
        peak_lag = correlations.index(max(correlations))
        fig_lag.add_annotation(
            x=peak_lag, y=max(correlations),
            text=f"Peak at {peak_lag} weeks",
            showarrow=True,
            arrowhead=2,
            font=dict(color=COLOR_SUCCESS)
        )
        
        st.plotly_chart(fig_lag, use_container_width=True, key="eda_lag_corr")
    
    # Regional breakdown
    if 'REGION' in df_weekly.columns:
        st.markdown("### Regional Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            df_reg = df_weekly.groupby(['WEEK_START', 'REGION'])['REVENUE'].sum().reset_index()
            fig_reg = px.line(
                df_reg, x='WEEK_START', y='REVENUE', color='REGION',
                title="Revenue by Region Over Time"
            )
            fig_reg = apply_plotly_theme(fig_reg)
            fig_reg.update_layout(height=350)
            st.plotly_chart(fig_reg, use_container_width=True, key="eda_regional")
        
        with col2:
            df_reg_total = df_weekly.groupby('REGION')[['SPEND', 'REVENUE']].sum().reset_index()
            fig_pie = px.pie(
                df_reg_total, values='REVENUE', names='REGION',
                title="Revenue Distribution by Region"
            )
            fig_pie = apply_plotly_theme(fig_pie)
            fig_pie.update_layout(height=350)
            st.plotly_chart(fig_pie, use_container_width=True, key="eda_pie")
    
    # Correlation analysis
    st.markdown("### Correlation Analysis")
    
    if 'CHANNEL' in df_weekly.columns:
        pivot_spend = df_weekly.pivot_table(
            index='WEEK_START', columns='CHANNEL', values='SPEND', aggfunc='sum'
        ).fillna(0)
    else:
        pivot_spend = df_weekly[['WEEK_START', 'SPEND']].set_index('WEEK_START')
    
    target = df_weekly.groupby('WEEK_START')['REVENUE'].sum()
    df_corr = pivot_spend.join(target).dropna()
    corr_matrix = df_corr.corr()
    
    if 'REVENUE' in corr_matrix.columns:
        rev_corr = corr_matrix['REVENUE'].drop('REVENUE').sort_values(ascending=False)
        
        fig_corr = go.Figure(go.Bar(
            x=rev_corr.values,
            y=rev_corr.index,
            orientation='h',
            marker_color=[COLOR_SUCCESS if v > 0 else COLOR_WARNING for v in rev_corr.values]
        ))
        fig_corr = apply_plotly_theme(fig_corr)
        fig_corr.update_layout(
            title="Correlation with Revenue (Raw Spend)",
            xaxis_title="Correlation Coefficient",
            height=max(300, len(rev_corr) * 25)
        )
        st.plotly_chart(fig_corr, use_container_width=True, key="eda_corr")


def render_diagnostics_tab(df_results: pd.DataFrame, df_weekly: pd.DataFrame):
    """Render Model Diagnostics tab with enhanced metrics."""
    if df_results.empty:
        st.info("No model results available. Run the training pipeline first.")
        return
    
    # Model Health Card
    render_model_health_card(df_results, df_weekly)
    
    # Educational panel
    with st.expander("Learn More: Time-Series Cross-Validation", expanded=False):
        exp = get_explanation("time_series_cv")
        st.markdown(exp.get("content", ""), unsafe_allow_html=True)
    
    st.markdown("### Channel Coefficients with Confidence Intervals")
    
    # Sort by ROI
    df_sorted = df_results.sort_values('ROI', ascending=True) if 'ROI' in df_results.columns else df_results
    
    # Create coefficient chart with error bars
    if 'ROI_CI_LOWER' in df_results.columns and 'ROI_CI_UPPER' in df_results.columns:
        fig_coeff = go.Figure()
        
        # Add bars with error bars
        fig_coeff.add_trace(go.Bar(
            x=df_sorted['ROI'],
            y=df_sorted['CHANNEL'],
            orientation='h',
            marker_color=[COLOR_SUCCESS if row.get('IS_SIGNIFICANT', True) else COLOR_WARNING 
                         for _, row in df_sorted.iterrows()],
            error_x=dict(
                type='data',
                array=(df_sorted['ROI_CI_UPPER'] - df_sorted['ROI']).tolist(),
                arrayminus=(df_sorted['ROI'] - df_sorted['ROI_CI_LOWER']).tolist(),
                color='rgba(255, 255, 255, 0.4)',
                thickness=1.5,
                width=4
            ),
            text=[f"{v:.2f}x" for v in df_sorted['ROI']],
            textposition='outside'
        ))
        
        fig_coeff = apply_plotly_theme(fig_coeff)
        fig_coeff.update_layout(
            title="ROI by Channel with 90% Confidence Intervals",
            xaxis_title="ROI",
            yaxis_title="",
            shapes=[
                dict(type='line', x0=1, x1=1, y0=-0.5, y1=len(df_sorted)-0.5,
                     line=dict(color='white', width=1, dash='dash'))
            ],
            annotations=[
                dict(x=1, y=len(df_sorted), text="Breakeven", showarrow=False,
                     font=dict(color='rgba(255,255,255,0.5)', size=9), yshift=10)
            ],
            height=max(350, len(df_sorted) * 30)
        )
        st.plotly_chart(fig_coeff, use_container_width=True, key="diag_coeff_ci")
    else:
        # Fallback without CI
        if 'COEFF_WEIGHT' in df_results.columns:
            df_coeff = df_results.sort_values('COEFF_WEIGHT', ascending=True)
            
            fig_coeff = go.Figure(go.Bar(
                x=df_coeff['COEFF_WEIGHT'],
                y=df_coeff['CHANNEL'],
                orientation='h',
                marker_color=COLOR_PRIMARY,
                text=[f"{v:.4f}" for v in df_coeff['COEFF_WEIGHT']],
                textposition='outside'
            ))
            fig_coeff = apply_plotly_theme(fig_coeff)
            fig_coeff.update_layout(
                title="Model Coefficient Weights by Channel",
                xaxis_title="Coefficient Weight",
                height=max(300, len(df_coeff) * 30)
            )
            st.plotly_chart(fig_coeff, use_container_width=True, key="diag_coeff")
    
    # Educational panel
    with st.expander("Learn More: Bootstrap Confidence Intervals", expanded=False):
        exp = get_explanation("bootstrap_ci")
        st.markdown(exp.get("content", ""), unsafe_allow_html=True)
    
    # Parameter Table
    st.markdown("### Learned MMM Parameters")
    st.markdown(
        "<p style='color: rgba(255,255,255,0.6);'>"
        "These parameters were optimized via Nevergrad to best explain the spend-revenue relationship."
        "</p>",
        unsafe_allow_html=True
    )
    
    # Build parameter table
    param_cols = ['CHANNEL', 'ROI', 'ROI_CI_LOWER', 'ROI_CI_UPPER', 'IS_SIGNIFICANT',
                  'ADSTOCK_DECAY', 'SATURATION_GAMMA', 'MARGINAL_ROI', 'N_OBSERVATIONS']
    available_cols = [c for c in param_cols if c in df_results.columns]
    
    if available_cols:
        df_params = df_results[available_cols].copy()
        
        # Format display
        if 'ROI' in df_params.columns:
            df_params['ROI'] = df_params['ROI'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        if 'ADSTOCK_DECAY' in df_params.columns:
            df_params['ADSTOCK_DECAY'] = df_params['ADSTOCK_DECAY'].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
            )
        if 'SATURATION_GAMMA' in df_params.columns:
            df_params['SATURATION_GAMMA'] = df_params['SATURATION_GAMMA'].apply(
                lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
            )
        if 'MARGINAL_ROI' in df_params.columns:
            df_params['MARGINAL_ROI'] = df_params['MARGINAL_ROI'].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
            )
        
        st.dataframe(
            df_params,
            use_container_width=True,
            hide_index=True,
            column_config={
                "CHANNEL": st.column_config.TextColumn("Channel"),
                "ROI": st.column_config.TextColumn("ROI"),
                "IS_SIGNIFICANT": st.column_config.CheckboxColumn("Significant"),
                "ADSTOCK_DECAY": st.column_config.TextColumn("θ (Decay)"),
                "SATURATION_GAMMA": st.column_config.TextColumn("γ (Half-Sat)"),
                "MARGINAL_ROI": st.column_config.TextColumn("Marginal ROI"),
                "N_OBSERVATIONS": st.column_config.NumberColumn("N Obs")
            }
        )
        
        # Download button
        csv = df_results.to_csv(index=False)
        st.download_button(
            "Export Parameters to CSV",
            csv,
            "mmm_parameters.csv",
            "text/csv"
        )
    
    # Educational panel
    with st.expander("Learn More: Nevergrad Optimization", expanded=False):
        exp = get_explanation("nevergrad_optimization")
        st.markdown(exp.get("content", ""), unsafe_allow_html=True)


def render_curves_tab(df_curves: pd.DataFrame, df_results: pd.DataFrame):
    """Render Response Curves tab with zones and CI bands."""
    if df_curves.empty:
        st.info("No response curves available. Run the training pipeline first.")
        return
    
    st.markdown("### Channel Response Curves")
    st.markdown(
        "<p style='color: rgba(255,255,255,0.6);'>"
        "Response curves show the relationship between spend and predicted revenue, "
        "illustrating diminishing returns. Shaded zones indicate efficiency levels."
        "</p>",
        unsafe_allow_html=True
    )
    
    channels = df_curves['CHANNEL'].unique().tolist()
    
    # Multi-select for channels to compare
    selected_channels = st.multiselect(
        "Select channels to compare",
        channels,
        default=channels[:3] if len(channels) >= 3 else channels,
        key="curve_select"
    )
    
    if not selected_channels:
        st.info("Select at least one channel to view response curves.")
        return
    
    # Get parameters for annotation
    channel_params = {}
    if not df_results.empty:
        for _, row in df_results.iterrows():
            ch = row.get('CHANNEL', '')
            channel_params[ch] = {
                'gamma': row.get('SATURATION_GAMMA', None),
                'marginal_roi': row.get('MARGINAL_ROI', None)
            }
    
    # Plot response curves
    fig = go.Figure()
    
    colors = [COLOR_PRIMARY, COLOR_ACCENT, COLOR_SUCCESS, COLOR_WARNING, '#9B59B6', '#1ABC9C']
    
    for i, ch in enumerate(selected_channels):
        ch_data = df_curves[df_curves['CHANNEL'] == ch].sort_values('SPEND')
        color = colors[i % len(colors)]
        
        # Add CI band if available
        if 'PREDICTED_REVENUE_CI_LOWER' in ch_data.columns:
            fig.add_trace(go.Scatter(
                x=ch_data['SPEND'].tolist() + ch_data['SPEND'].tolist()[::-1],
                y=ch_data['PREDICTED_REVENUE_CI_UPPER'].fillna(ch_data['PREDICTED_REVENUE'] * 1.15).tolist() + 
                  ch_data['PREDICTED_REVENUE_CI_LOWER'].fillna(ch_data['PREDICTED_REVENUE'] * 0.85).tolist()[::-1],
                fill='toself',
                fillcolor=f'rgba{tuple(list(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.1])}',
                line=dict(width=0),
                name=f'{ch} 90% CI',
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Main curve
        fig.add_trace(go.Scatter(
            x=ch_data['SPEND'],
            y=ch_data['PREDICTED_REVENUE'],
            mode='lines',
            name=ch,
            line=dict(color=color, width=2)
        ))
    
    fig = apply_plotly_theme(fig)
    fig.update_layout(
        title="Saturation Curves: Spend vs Predicted Revenue",
        xaxis_title="Spend ($)",
        yaxis_title="Predicted Revenue ($)",
        legend=dict(orientation='h', y=1.15),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True, key="curves_main")
    
    # ==========================================================================
    # LAYER 2: Quantitative Metrics Panel for Response Curves
    # ==========================================================================
    st.markdown("#### Response Curve Diagnostics")
    st.markdown(
        "<p style='color: rgba(255,255,255,0.5); font-size: 0.85rem;'>"
        "Key metrics for understanding channel saturation and efficiency</p>",
        unsafe_allow_html=True
    )
    
    # Calculate and display metrics for each selected channel
    response_curve_metrics = {}
    
    for ch in selected_channels:
        ch_data = df_curves[df_curves['CHANNEL'] == ch].sort_values('SPEND')
        ch_params = channel_params.get(ch, {})
        
        if ch_data.empty:
            continue
        
        # Calculate metrics
        current_spend = float(ch_data['SPEND'].mean())  # Mean as proxy for "current"
        max_spend = float(ch_data['SPEND'].max())
        min_spend = float(ch_data['SPEND'].min())
        
        # Get gamma (half-saturation) from params or estimate from data
        gamma = ch_params.get('gamma')
        if gamma is None:
            gamma = float(ch_data['SPEND'].median())
        
        # Saturation percentage: how far along the curve (0% = at min, 100% = at 2x gamma)
        saturation_pct = min(100, max(0, (current_spend / (gamma * 2)) * 100)) if gamma > 0 else 50
        
        # Calculate marginal ROI at current spend level
        spends = ch_data['SPEND'].values
        revenues = ch_data['PREDICTED_REVENUE'].values
        if len(spends) > 1:
            marginal_values = np.gradient(revenues, spends)
            marginal_at_current = float(np.interp(current_spend, spends, marginal_values))
        else:
            marginal_at_current = ch_params.get('marginal_roi', 1.0) or 1.0
        
        # Optimal range: 0.5x to 1.5x gamma
        optimal_lower = gamma * 0.5
        optimal_upper = gamma * 1.5
        
        # Headroom: how much more can be spent before deep saturation
        headroom = max(0, optimal_upper - current_spend)
        
        # Store for AI narrative
        response_curve_metrics[ch] = {
            'current_spend': current_spend,
            'saturation_pct': saturation_pct,
            'marginal_roi': marginal_at_current,
            'optimal_lower': optimal_lower,
            'optimal_upper': optimal_upper,
            'headroom': headroom,
            'gamma': gamma
        }
        
        # Determine efficiency zone
        if marginal_at_current > 1.5:
            zone = "EFFICIENT"
            zone_color = COLOR_SUCCESS
        elif marginal_at_current >= 0.8:
            zone = "DIMINISHING"
            zone_color = COLOR_WARNING
        else:
            zone = "SATURATED"
            zone_color = COLOR_DANGER
        
        # Display metrics in expander per channel
        with st.expander(f"{ch} - {zone}", expanded=(len(selected_channels) == 1)):
            cols = st.columns(5)
            cols[0].metric(
                "Current Spend",
                f"${current_spend:,.0f}",
                help="Average weekly spend level for this channel"
            )
            cols[1].metric(
                "Saturation",
                f"{saturation_pct:.0f}%",
                help="How close to full saturation (0%=efficient, 100%=saturated)"
            )
            cols[2].metric(
                "Marginal ROI",
                f"{marginal_at_current:.2f}x",
                help="Revenue generated per additional dollar at current spend",
                delta=f"{'Above' if marginal_at_current >= 1.0 else 'Below'} breakeven",
                delta_color="normal" if marginal_at_current >= 1.0 else "inverse"
            )
            cols[3].metric(
                "Optimal Range",
                f"${optimal_lower:,.0f} - ${optimal_upper:,.0f}",
                help="Recommended spend range for best efficiency"
            )
            cols[4].metric(
                "Headroom",
                f"${headroom:,.0f}",
                help="Additional spend available before deep saturation",
                delta="Room to grow" if headroom > 10000 else "Near limit",
                delta_color="normal" if headroom > 10000 else "inverse"
            )
            
            # Zone indicator
            st.markdown(
                f"<p style='margin-top: 0.5rem;'>"
                f"<strong>Efficiency Zone:</strong> "
                f"<span style='color: {zone_color}; font-weight: 600;'>{zone}</span> "
                f"- {'High returns on additional spend' if zone == 'EFFICIENT' else 'Moderate returns, optimize carefully' if zone == 'DIMINISHING' else 'Low returns, consider reallocation'}"
                f"</p>",
                unsafe_allow_html=True
            )
    
    # Educational panel
    with st.expander("How to interpret these metrics", expanded=False):
        exp = get_explanation("saturation_curves")
        st.markdown(exp.get("content", ""), unsafe_allow_html=True)
    
    # ==========================================================================
    # LAYER 3: AI Narrative for Response Curves
    # ==========================================================================
    if response_curve_metrics and len(selected_channels) > 0:
        st.markdown("#### AI Interpretation")
        
        # Generate narrative for the primary selected channel
        primary_channel = selected_channels[0]
        primary_metrics = response_curve_metrics.get(primary_channel, {})
        
        if primary_metrics:
            # Try to get session for AI generation
            try:
                session = get_active_session()
                
                with st.spinner("Generating AI interpretation..."):
                    narrative = generate_diagnostic_narrative(
                        session,
                        context_type="response_curve",
                        channel=primary_channel,
                        metrics=primary_metrics
                    )
                
                if narrative:
                    st.markdown(f"""
                    <div style="background: linear-gradient(145deg, rgba(26, 31, 46, 0.8) 0%, rgba(37, 45, 61, 0.6) 100%);
                                border: 1px solid rgba(41, 181, 232, 0.15); border-radius: 12px; padding: 1.25rem; margin: 1rem 0;">
                        <div style="display: flex; align-items: center; gap: 0.5rem; color: {COLOR_PRIMARY}; 
                                    font-weight: 600; font-size: 0.95rem; margin-bottom: 0.75rem;">
                            AI Analysis for {primary_channel}
                        </div>
                        <div style="color: rgba(255, 255, 255, 0.85); font-size: 0.95rem; line-height: 1.6;">
                            {narrative}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("AI interpretation unavailable. Ensure Cortex is configured.")
                    
            except Exception as e:
                # Fallback: show a template-based interpretation
                zone = "efficient" if primary_metrics.get('marginal_roi', 0) > 1.5 else \
                       "diminishing returns" if primary_metrics.get('marginal_roi', 0) >= 0.8 else "saturated"
                action = "consider increasing spend" if zone == "efficient" else \
                         "optimize carefully" if zone == "diminishing returns" else "consider reallocating budget"
                
                st.markdown(f"""
                <div style="background: {BG_CARD}; border-radius: 12px; padding: 1rem; margin: 0.5rem 0;">
                    <strong>{primary_channel}</strong> is currently in the <strong>{zone}</strong> zone 
                    with a marginal ROI of {primary_metrics.get('marginal_roi', 0):.2f}x. 
                    Based on current saturation ({primary_metrics.get('saturation_pct', 0):.0f}%), 
                    you should {action}.
                </div>
                """, unsafe_allow_html=True)
        
        # If multiple channels selected, show comparative narrative
        if len(selected_channels) > 1 and len(response_curve_metrics) > 1:
            try:
                session = get_active_session()
                
                with st.spinner("Generating comparative analysis..."):
                    comparative = generate_comparative_narrative(
                        session,
                        selected_channels,
                        response_curve_metrics
                    )
                
                if comparative:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(41, 181, 232, 0.1) 0%, rgba(17, 86, 127, 0.1) 100%);
                                border: 1px solid {COLOR_PRIMARY}; border-radius: 12px; padding: 1.25rem; margin: 1rem 0;">
                        <div style="display: flex; align-items: center; gap: 0.5rem; color: {COLOR_PRIMARY}; 
                                    font-weight: 600; font-size: 0.95rem; margin-bottom: 0.75rem;">
                            Budget Reallocation Recommendation
                        </div>
                        <div style="color: rgba(255, 255, 255, 0.85); font-size: 0.95rem; line-height: 1.6;">
                            {comparative}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception:
                pass  # Silently skip comparative if unavailable
    
    # Marginal ROI curves
    st.markdown("### Marginal Efficiency")
    st.markdown(
        "<p style='color: rgba(255,255,255,0.6);'>"
        "Marginal efficiency shows the incremental revenue generated per additional dollar spent. "
        "The horizontal dashed line at 1.0 represents breakeven."
        "</p>",
        unsafe_allow_html=True
    )
    
    # Calculate marginal ROI from curves
    if 'MARGINAL_ROI_AT_SPEND' in df_curves.columns:
        fig_marginal = go.Figure()
        
        # Collect all mROI values for y-axis range calculation
        all_mroi_values = []
        
        for i, ch in enumerate(selected_channels):
            ch_data = df_curves[df_curves['CHANNEL'] == ch].sort_values('SPEND')
            color = colors[i % len(colors)]
            
            mroi_values = ch_data['MARGINAL_ROI_AT_SPEND'].values
            all_mroi_values.extend(mroi_values)
            
            fig_marginal.add_trace(go.Scatter(
                x=ch_data['SPEND'],
                y=mroi_values,
                mode='lines',
                name=ch,
                line=dict(color=color, width=2)
            ))
        
        fig_marginal = apply_plotly_theme(fig_marginal)
        
        # Calculate reasonable y-axis range to avoid extreme values making chart unreadable
        all_mroi = np.array(all_mroi_values)
        y_max = min(5.0, max(3.0, np.percentile(all_mroi[all_mroi > 0], 95) if (all_mroi > 0).any() else 3.0))
        y_min = max(-1.0, min(0, np.percentile(all_mroi, 5)))
        
        fig_marginal.update_layout(
            title="",
            xaxis_title="Spend ($)",
            yaxis_title="Marginal ROI ($ revenue per $ spend)",
            yaxis=dict(range=[y_min, y_max]),
            shapes=[
                dict(type='line', x0=0, x1=df_curves['SPEND'].max(),
                     y0=1, y1=1, line=dict(color='white', width=1, dash='dash'))
            ],
            legend=dict(orientation='h', y=1.15),
            height=400
        )
        
        # Add zone shading (adjusted to fit within y-axis range)
        fig_marginal.add_hrect(y0=1.5, y1=y_max, fillcolor="rgba(46, 204, 113, 0.1)",
                               layer="below", line_width=0,
                               annotation_text="Efficient", annotation_position="top left")
        fig_marginal.add_hrect(y0=0.8, y1=1.5, fillcolor="rgba(243, 156, 18, 0.1)",
                               layer="below", line_width=0,
                               annotation_text="Diminishing", annotation_position="top left")
        fig_marginal.add_hrect(y0=y_min, y1=0.8, fillcolor="rgba(231, 76, 60, 0.1)",
                               layer="below", line_width=0,
                               annotation_text="Saturated", annotation_position="top left")
        
        st.plotly_chart(fig_marginal, use_container_width=True, key="curves_marginal")
    else:
        # Calculate from response curves
        marginal_data = []
        
        for ch in selected_channels:
            ch_data = df_curves[df_curves['CHANNEL'] == ch].sort_values('SPEND')
            
            if len(ch_data) > 1:
                spends = ch_data['SPEND'].values
                # Floor revenue at 0 to avoid extreme negative gradients
                revenues = np.maximum(0, ch_data['PREDICTED_REVENUE'].values)
                
                marginal = np.gradient(revenues, spends)
                
                for s, m in zip(spends, marginal):
                    marginal_data.append({
                        'Channel': ch,
                        'Spend': s,
                        'Marginal ROI': m
                    })
        
        if marginal_data:
            df_marginal = pd.DataFrame(marginal_data)
            
            fig_marginal = go.Figure()
            
            for i, ch in enumerate(selected_channels):
                ch_data = df_marginal[df_marginal['Channel'] == ch]
                fig_marginal.add_trace(go.Scatter(
                    x=ch_data['Spend'],
                    y=ch_data['Marginal ROI'],
                    mode='lines',
                    name=ch,
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
            
            fig_marginal = apply_plotly_theme(fig_marginal)
            
            # Calculate reasonable y-axis range
            all_mroi = df_marginal['Marginal ROI'].values
            y_max = min(5.0, max(2.0, np.percentile(all_mroi[all_mroi > 0], 95) if (all_mroi > 0).any() else 2.0))
            y_min = max(-1.0, min(-0.5, np.percentile(all_mroi, 5)))
            
            fig_marginal.update_layout(
                title="",
                xaxis_title="Spend ($)",
                yaxis_title="Marginal ROI ($ revenue per $ spend)",
                yaxis=dict(range=[y_min, y_max]),
                legend=dict(orientation='h', y=1.15),
                height=400,
                shapes=[
                    dict(type='line', x0=0, x1=df_marginal['Spend'].max(),
                         y0=1, y1=1, line=dict(color='white', width=1, dash='dash'))
                ],
                annotations=[
                    dict(x=df_marginal['Spend'].max() * 0.95, y=1.05,
                         text="Breakeven (1.0x)", showarrow=False,
                         font=dict(color='rgba(255,255,255,0.6)', size=10))
                ]
            )
            st.plotly_chart(fig_marginal, use_container_width=True, key="curves_marginal")
            
            # ==========================================================================
            # LAYER 2: Quantitative Metrics Panel for Marginal Efficiency
            # ==========================================================================
            st.markdown("#### Marginal Efficiency Diagnostics")
            st.markdown(
                "<p style='color: rgba(255,255,255,0.5); font-size: 0.85rem;'>"
                "Key metrics for understanding where each channel stands on the efficiency curve</p>",
                unsafe_allow_html=True
            )
            
            # Calculate marginal efficiency metrics
            marginal_metrics = {}
            
            for idx, ch in enumerate(selected_channels):
                ch_marginal = df_marginal[df_marginal['Channel'] == ch]
                
                if ch_marginal.empty:
                    continue
                
                # Calculate key metrics
                mroi_values = ch_marginal['Marginal ROI'].values
                spend_values = ch_marginal['Spend'].values
                
                # Current mROI (at midpoint of spend range)
                mid_idx = len(mroi_values) // 2
                current_mroi = float(mroi_values[mid_idx]) if len(mroi_values) > 0 else 1.0
                
                # Peak efficiency point
                peak_idx = np.argmax(mroi_values)
                peak_mroi = float(mroi_values[peak_idx])
                peak_spend = float(spend_values[peak_idx])
                
                # Breakeven point (where mROI crosses 1.0)
                above_one = mroi_values >= 1.0
                if above_one.any() and not above_one.all():
                    # Find first crossing below 1.0
                    crossings = np.where(np.diff(above_one.astype(int)) != 0)[0]
                    if len(crossings) > 0:
                        breakeven_spend = float(spend_values[crossings[0]])
                    else:
                        breakeven_spend = float(spend_values[-1])
                elif above_one.all():
                    breakeven_spend = float(spend_values[-1])  # Never breaks even in range
                else:
                    breakeven_spend = float(spend_values[0])  # Already below breakeven
                
                marginal_metrics[ch] = {
                    'current_mroi': current_mroi,
                    'peak_mroi': peak_mroi,
                    'peak_spend': peak_spend,
                    'breakeven_spend': breakeven_spend,
                    'rank': idx + 1
                }
                
                # Determine recommendation
                if current_mroi > 1.5:
                    recommendation = "Increase"
                    rec_color = COLOR_SUCCESS
                elif current_mroi >= 0.8:
                    recommendation = "Maintain"
                    rec_color = COLOR_WARNING
                else:
                    recommendation = "Decrease"
                    rec_color = COLOR_DANGER
                
                # Display in expander
                with st.expander(f"{ch} - {recommendation} Spend", expanded=(len(selected_channels) == 1)):
                    cols = st.columns(4)
                    cols[0].metric(
                        "Current mROI",
                        f"{current_mroi:.2f}x",
                        help="Marginal ROI at current spend level",
                        delta="Profitable" if current_mroi >= 1.0 else "Below breakeven",
                        delta_color="normal" if current_mroi >= 1.0 else "inverse"
                    )
                    cols[1].metric(
                        "Peak Efficiency",
                        f"${peak_spend:,.0f}",
                        help=f"Spend level where mROI peaks at {peak_mroi:.2f}x"
                    )
                    cols[2].metric(
                        "Breakeven Point",
                        f"${breakeven_spend:,.0f}",
                        help="Spend level where marginal ROI = 1.0x"
                    )
                    cols[3].metric(
                        "Recommendation",
                        recommendation,
                        help="Budget action based on current marginal efficiency"
                    )
                    
                    # Recommendation text
                    st.markdown(
                        f"<p style='color: {rec_color}; font-weight: 500;'>"
                        f"{'Room to grow - each dollar returns ${:.2f}'.format(current_mroi) if current_mroi > 1.2 else 'Near optimal - maintain current levels' if current_mroi >= 0.9 else 'Diminishing returns - consider reallocation'}"
                        f"</p>",
                        unsafe_allow_html=True
                    )
            
            # ==========================================================================
            # LAYER 3: AI Narrative for Marginal Efficiency
            # ==========================================================================
            if marginal_metrics and len(selected_channels) > 0:
                st.markdown("#### AI Interpretation")
                
                primary_channel = selected_channels[0]
                primary_mmetrics = marginal_metrics.get(primary_channel, {})
                
                if primary_mmetrics:
                    try:
                        session = get_active_session()
                        
                        with st.spinner("Generating AI interpretation..."):
                            m_narrative = generate_diagnostic_narrative(
                                session,
                                context_type="marginal_efficiency",
                                channel=primary_channel,
                                metrics=primary_mmetrics
                            )
                        
                        if m_narrative:
                            st.markdown(f"""
                            <div style="background: linear-gradient(145deg, rgba(26, 31, 46, 0.8) 0%, rgba(37, 45, 61, 0.6) 100%);
                                        border: 1px solid rgba(41, 181, 232, 0.15); border-radius: 12px; padding: 1.25rem; margin: 1rem 0;">
                                <div style="display: flex; align-items: center; gap: 0.5rem; color: {COLOR_PRIMARY}; 
                                            font-weight: 600; font-size: 0.95rem; margin-bottom: 0.75rem;">
                                    AI Budget Recommendation for {primary_channel}
                                </div>
                                <div style="color: rgba(255, 255, 255, 0.85); font-size: 0.95rem; line-height: 1.6;">
                                    {m_narrative}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("AI interpretation unavailable. Ensure Cortex is configured.")
                            
                    except Exception:
                        # Fallback template
                        rec = "increase spend" if primary_mmetrics.get('current_mroi', 0) > 1.2 else \
                              "maintain current levels" if primary_mmetrics.get('current_mroi', 0) >= 0.9 else \
                              "consider reducing spend"
                        st.markdown(f"""
                        <div style="background: {BG_CARD}; border-radius: 12px; padding: 1rem; margin: 0.5rem 0;">
                            Based on marginal efficiency analysis, <strong>{primary_channel}</strong> 
                            has a current marginal ROI of {primary_mmetrics.get('current_mroi', 0):.2f}x. 
                            Recommendation: {rec}.
                        </div>
                        """, unsafe_allow_html=True)
    
    # Educational panel
    with st.expander("Learn More: Hill Saturation Function", expanded=False):
        exp = get_explanation("hill_function")
        st.markdown(exp.get("content", ""), unsafe_allow_html=True)


def render_analyst_tab(session):
    """Render Cortex Analyst tab."""
    st.markdown("### Cortex Analyst")
    st.markdown(
        "<p style='color: rgba(255,255,255,0.6);'>"
        "Ask natural language questions about your marketing data. "
        "Cortex Analyst will generate and execute SQL queries automatically."
        "</p>",
        unsafe_allow_html=True
    )
    
    # Example queries
    render_example_queries(key_prefix="explorer_analyst")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if "explorer_analyst_suggested_query" in st.session_state:
        suggested = st.session_state.pop("explorer_analyst_suggested_query")
        st.info(f"Try asking: \"{suggested}\"")
    
    render_analyst_chat(session, key_prefix="explorer_analyst")


def main():
    # --- Session & Data ---
    try:
        session = get_active_session()
    except Exception:
        st.error("Could not connect to Snowflake. Please ensure you're running in Snowflake.")
        return
    
    with st.spinner("Loading model data..."):
        data = load_explorer_data(session)
        df_weekly = data.get("WEEKLY", pd.DataFrame())
        df_curves = data.get("CURVES", pd.DataFrame())
        df_results = data.get("RESULTS", pd.DataFrame())
        df_roi = data.get("ROI", pd.DataFrame())
        df_metadata = data.get("METADATA", pd.DataFrame())

    # --- Header ---
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1>Model Explorer</h1>
        <p style="color: rgba(255,255,255,0.6); font-size: 1.1rem;">
            Technical deep-dive into MMM diagnostics, learned parameters, and ad-hoc analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model version info
    if not df_metadata.empty:
        version = df_metadata['MODEL_VERSION'].iloc[0] if 'MODEL_VERSION' in df_metadata.columns else "Unknown"
        run_date = df_metadata['MODEL_RUN_DATE'].iloc[0] if 'MODEL_RUN_DATE' in df_metadata.columns else "Unknown"
        st.markdown(f"""
        <div style="background: {BG_CARD}; border-radius: 8px; padding: 0.5rem 1rem; 
                    display: inline-flex; gap: 1rem; margin-bottom: 1rem;">
            <span style="color: rgba(255,255,255,0.5);">Model Version:</span>
            <span style="color: {COLOR_PRIMARY}; font-weight: 600;">{version}</span>
            <span style="color: rgba(255,255,255,0.3);">|</span>
            <span style="color: rgba(255,255,255,0.5);">Last Run:</span>
            <span style="color: white;">{run_date}</span>
        </div>
        """, unsafe_allow_html=True)

    # --- Tabs ---
    tab_diag, tab_eda, tab_curves, tab_analyst = st.tabs([
        "Model Diagnostics",
        "Exploratory Analysis",
        "Response Curves",
        "Cortex Analyst"
    ])
    
    with tab_diag:
        render_diagnostics_tab(df_results, df_weekly)
    
    with tab_eda:
        render_eda_tab(df_weekly)
    
    with tab_curves:
        render_curves_tab(df_curves, df_results)
    
    with tab_analyst:
        render_analyst_tab(session)

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
        if st.button("Budget Simulator", use_container_width=True):
            st.switch_page("pages/2_Simulator.py")


if __name__ == "__main__":
    main()
