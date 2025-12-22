"""
Regional Map Visualization for MMM ROI Engine.

Provides geographic visualization of ROI metrics by region with drill-down capability.
Uses PyDeck for Snowflake Streamlit compatibility (Plotly Geo is blocked by CSP).
"""
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

from utils.styling import (
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_ACCENT,
    COLOR_SUCCESS,
    COLOR_WARNING,
    COLOR_DANGER,
    BG_CARD,
)


# =============================================================================
# Region Configuration
# =============================================================================

REGION_COORDS = {
    'NA': {
        'lat': 39.8,
        'lon': -98.5,
        'name': 'North America',
        'full_name': 'North America'
    },
    'EMEA': {
        'lat': 48.8,
        'lon': 10.0,
        'name': 'EMEA',
        'full_name': 'Europe, Middle East & Africa'
    },
    'APAC': {
        'lat': 25.0,
        'lon': 115.0,
        'name': 'APAC',
        'full_name': 'Asia Pacific'
    },
    'LATAM': {
        'lat': -15.0,
        'lon': -60.0,
        'name': 'LATAM',
        'full_name': 'Latin America'
    },
}


def get_roi_color(roi: float) -> str:
    """Get hex color based on ROI value."""
    if roi > 1.5:
        return COLOR_SUCCESS
    elif roi >= 0.5:
        return COLOR_WARNING
    else:
        return COLOR_DANGER


def get_roi_color_rgba(roi: float) -> list:
    """Get RGBA color array for PyDeck based on ROI value."""
    if roi > 1.5:
        return [16, 185, 129, 200]   # Green (#10b981)
    elif roi >= 0.5:
        return [245, 158, 11, 200]   # Amber (#f59e0b)
    else:
        return [239, 68, 68, 200]    # Red (#ef4444)


def extract_region_from_channel(channel_name: str) -> str:
    """Extract region code from channel name like 'Facebook_NA_ALL'."""
    parts = channel_name.split('_')
    if len(parts) >= 3:
        return parts[-2]  # NA, EMEA, APAC, LATAM
    elif len(parts) == 2:
        return parts[-1]
    return 'Global'


def extract_base_channel(channel_name: str) -> str:
    """Extract base channel from channel name like 'Facebook_NA_ALL'."""
    parts = channel_name.split('_')
    if len(parts) >= 2:
        return parts[0]  # Facebook, LinkedIn, Google Ads, Programmatic
    return channel_name


def aggregate_by_region(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate model results by region.
    
    Returns DataFrame with columns:
    - REGION: Region code (NA, EMEA, APAC, LATAM)
    - AVG_ROI: Average ROI across channels in region
    - TOTAL_SPEND: Sum of current spend
    - TOTAL_REVENUE: Estimated attributed revenue
    - CHANNEL_COUNT: Number of channels
    - CHANNELS: List of channel names
    """
    if df_results.empty or 'CHANNEL' not in df_results.columns:
        return pd.DataFrame()
    
    df = df_results.copy()
    df['REGION'] = df['CHANNEL'].apply(extract_region_from_channel)
    df['BASE_CHANNEL'] = df['CHANNEL'].apply(extract_base_channel)
    
    # Aggregate by region
    agg = df.groupby('REGION').agg({
        'ROI': 'mean',
        'CURRENT_SPEND': 'sum',
        'MARGINAL_ROI': 'mean',
        'CHANNEL': lambda x: list(x)
    }).reset_index()
    
    agg.columns = ['REGION', 'AVG_ROI', 'TOTAL_SPEND', 'AVG_MARGINAL_ROI', 'CHANNELS']
    agg['CHANNEL_COUNT'] = agg['CHANNELS'].apply(len)
    
    # Estimate revenue (spend * ROI) - show actual value including negative for transparency
    agg['ESTIMATED_REVENUE'] = agg['TOTAL_SPEND'] * agg['AVG_ROI']
    
    return agg


def prepare_regional_map_data(df_results: pd.DataFrame, 
                              selected_region: str = None) -> pd.DataFrame:
    """
    Prepare regional data for PyDeck map visualization.
    
    Args:
        df_results: Model results DataFrame with CHANNEL, ROI, CURRENT_SPEND
        selected_region: Currently selected region to highlight (optional)
    
    Returns:
        DataFrame with lat, lon, colors, and pre-formatted tooltip values
    """
    # Aggregate by region
    df_regional = aggregate_by_region(df_results)
    
    if df_regional.empty:
        return pd.DataFrame()
    
    # Build map data
    map_data = []
    for _, row in df_regional.iterrows():
        region = row['REGION']
        if region not in REGION_COORDS:
            continue
            
        coords = REGION_COORDS[region]
        roi = row['AVG_ROI']
        spend = row['TOTAL_SPEND']
        is_selected = region == selected_region
        
        # Scale radius by spend (PyDeck uses meters, so scale appropriately)
        base_radius = max(300000, min(800000, np.sqrt(spend / 1e6) * 100000))
        # Increase size for selected region
        radius = base_radius * 1.3 if is_selected else base_radius
        
        map_data.append({
            'region': region,
            'region_name': coords['full_name'],
            'lat': coords['lat'],
            'lon': coords['lon'],
            'roi': roi,
            'spend': spend,
            'channel_count': row['CHANNEL_COUNT'],
            'color': get_roi_color_rgba(roi),
            'radius': radius,
            'is_selected': is_selected,
            # Pre-formatted strings for tooltips (PyDeck doesn't support format specifiers)
            'roi_display': f"{roi:.2f}x",
            'spend_display': f"${spend:,.0f}",
            'channel_count_display': str(int(row['CHANNEL_COUNT']))
        })
    
    return pd.DataFrame(map_data)


def render_segmented_control(options: list, 
                             current_value: str,
                             key_prefix: str = "seg") -> str:
    """
    Render a segmented control (button group) for selection.
    
    Args:
        options: List of dicts with 'value', 'label', and optional 'badge' keys
        current_value: Currently selected value
        key_prefix: Unique key prefix for session state
        
    Returns:
        Selected value string
    """
    # Build the segmented control HTML
    buttons_html = []
    for opt in options:
        value = opt['value']
        label = opt['label']
        badge = opt.get('badge', '')
        is_active = value == current_value
        active_class = 'active' if is_active else ''
        
        badge_html = f'<span class="roi-badge">({badge})</span>' if badge else ''
        buttons_html.append(
            f'<span class="seg-btn {active_class}" data-value="{value}">'
            f'{label}{badge_html}</span>'
        )
    
    # Display the styled segmented control
    st.markdown(
        '<p style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin-bottom: 0.5rem;">Filter by Region:</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="segmented-control">{"".join(buttons_html)}</div>',
        unsafe_allow_html=True
    )
    
    # Use columns with buttons for actual interactivity
    num_options = len(options)
    cols = st.columns(num_options)
    
    for idx, opt in enumerate(options):
        with cols[idx]:
            value = opt['value']
            label = opt['label']
            badge = opt.get('badge', '')
            display_text = f"{label} ({badge})" if badge else label
            is_selected = value == current_value
            
            # Use button with custom styling via key
            if st.button(
                display_text,
                key=f"{key_prefix}_btn_{value}",
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                return value
    
    return current_value


def render_region_selector_map(df_results: pd.DataFrame,
                               key_prefix: str = "map") -> str:
    """
    Render interactive map with region selection via segmented control.
    Uses PyDeck for Snowflake Streamlit compatibility.
    
    Returns the selected region code or None for "All Regions".
    """
    # Aggregate data
    df_regional = aggregate_by_region(df_results)
    
    # Initialize session state
    state_key = f"{key_prefix}_selected_region"
    if state_key not in st.session_state:
        st.session_state[state_key] = None
    
    # Prepare map data
    df_map = prepare_regional_map_data(df_results, st.session_state[state_key])
    
    if not df_map.empty:
        # Create PyDeck ScatterplotLayer for regional bubbles
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=df_map,
            get_position='[lon, lat]',
            get_radius='radius',
            get_fill_color='color',
            pickable=True,
            opacity=0.85,
            stroked=True,
            get_line_color=[255, 255, 255, 150],
            line_width_min_pixels=2,
        )
        
        # World view centered to show all regions
        view_state = pdk.ViewState(
            latitude=25,
            longitude=10,
            zoom=1.2,
            pitch=0,
            min_zoom=0.8,
            max_zoom=6
        )
        
        # Create deck with map_style=None (CRITICAL for Snowflake Streamlit)
        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={
                'html': '<b>{region_name}</b><br>'
                        'ROI: {roi_display}<br>'
                        'Spend: {spend_display}<br>'
                        'Channels: {channel_count_display}',
                'style': {
                    'backgroundColor': '#1e293b',
                    'color': '#e2e8f0',
                    'borderRadius': '8px',
                    'padding': '8px',
                    'fontSize': '12px'
                }
            },
            map_style=None  # CRITICAL - lets Snowflake inject proper Mapbox/Carto token
        )
        
        st.pydeck_chart(deck, use_container_width=True)
        
        # Add legend for ROI colors
        st.markdown("""
        <div style="display: flex; gap: 1.5rem; justify-content: center; margin-top: 0.5rem; margin-bottom: 1rem;">
            <span style="color: #94a3b8;"><span style="color: #10b981; font-weight: bold;">[High]</span> High ROI (&gt;1.5x)</span>
            <span style="color: #94a3b8;"><span style="color: #f59e0b; font-weight: bold;">[Medium]</span> Medium ROI (0.5-1.5x)</span>
            <span style="color: #94a3b8;"><span style="color: #ef4444; font-weight: bold;">[Low]</span> Low ROI (&lt;0.5x)</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No regional data available for map visualization.")
    
    # Build segmented control options with ROI values
    seg_options = [{'value': 'all', 'label': 'All Regions', 'badge': ''}]
    
    for region_code in REGION_COORDS.keys():
        region_data = df_regional[df_regional['REGION'] == region_code]
        if not region_data.empty:
            roi = region_data['AVG_ROI'].iloc[0]
            roi_text = f"{roi:.1f}x"
        else:
            roi_text = "N/A"
        
        seg_options.append({
            'value': region_code,
            'label': region_code,
            'badge': roi_text
        })
    
    # Current selection value for segmented control
    current_sel = st.session_state[state_key] if st.session_state[state_key] else 'all'
    
    # Render segmented control with buttons
    st.markdown(
        '<p style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin-bottom: 0.5rem;">Filter by Region:</p>',
        unsafe_allow_html=True
    )
    
    # Create button columns for the segmented control
    cols = st.columns(len(seg_options))
    selected_value = current_sel
    
    for idx, opt in enumerate(seg_options):
        with cols[idx]:
            value = opt['value']
            label = opt['label']
            badge = opt.get('badge', '')
            display_text = f"{label} ({badge})" if badge else label
            is_selected = value == current_sel
            
            if st.button(
                display_text,
                key=f"{key_prefix}_seg_{value}",
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                selected_value = value
    
    # Convert 'all' back to None for the return value
    new_region = None if selected_value == 'all' else selected_value
    
    # Update session state if selection changed
    if new_region != st.session_state[state_key]:
        st.session_state[state_key] = new_region
        st.rerun()
    
    return st.session_state[state_key]


def render_region_drill_down(region: str, 
                            df_results: pd.DataFrame,
                            key_prefix: str = "drill") -> None:
    """
    Render channel breakdown cards for the selected region.
    
    Shows channel-level metrics in a grid layout.
    """
    if df_results.empty:
        st.info("No channel data available for this region.")
        return
    
    # Filter to selected region
    df = df_results.copy()
    df['REGION'] = df['CHANNEL'].apply(extract_region_from_channel)
    df['BASE_CHANNEL'] = df['CHANNEL'].apply(extract_base_channel)
    
    if region:
        df_filtered = df[df['REGION'] == region]
        region_name = REGION_COORDS.get(region, {}).get('full_name', region)
    else:
        df_filtered = df
        region_name = "All Regions"
    
    if df_filtered.empty:
        st.info(f"No channels found for {region_name}.")
        return
    
    # Sort by ROI descending
    df_filtered = df_filtered.sort_values('ROI', ascending=False)
    
    st.markdown(f"### {region_name} Channel Breakdown")
    st.markdown(
        f"<p style='color: rgba(255,255,255,0.6);'>"
        f"Showing {len(df_filtered)} channel{'s' if len(df_filtered) != 1 else ''}</p>",
        unsafe_allow_html=True
    )
    
    # Render channel cards in grid
    num_cols = min(4, len(df_filtered))
    cols = st.columns(num_cols)
    
    for idx, (_, row) in enumerate(df_filtered.iterrows()):
        col_idx = idx % num_cols
        
        with cols[col_idx]:
            roi = row.get('ROI', 0)
            spend = row.get('CURRENT_SPEND', 0)
            mroi = row.get('MARGINAL_ROI', 0)
            base_channel = row['BASE_CHANNEL']
            
            roi_color = get_roi_color(roi)
            
            # Determine efficiency zone
            if mroi > 1.5:
                zone = "EFFICIENT"
                zone_color = COLOR_SUCCESS
            elif mroi >= 0.8:
                zone = "DIMINISHING"
                zone_color = COLOR_WARNING
            else:
                zone = "SATURATED"
                zone_color = COLOR_DANGER
            
            st.markdown(f"""
            <div style="background: {BG_CARD}; border-radius: 12px; padding: 1rem; margin-bottom: 1rem;
                        border-left: 4px solid {roi_color};">
                <div style="font-weight: 600; color: white; margin-bottom: 0.5rem;">
                    {base_channel}
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: {roi_color};">
                            {roi:.2f}x
                        </div>
                        <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem;">ROI</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="color: white; font-weight: 500;">${spend/1e6:.1f}M</div>
                        <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem;">Spend</div>
                    </div>
                </div>
                <div style="margin-top: 0.5rem; display: flex; justify-content: space-between;">
                    <span style="color: {zone_color}; font-size: 0.75rem; font-weight: 500;">
                        {zone}
                    </span>
                    <span style="color: rgba(255,255,255,0.5); font-size: 0.75rem;">
                        mROI: {mroi:.2f}x
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_regional_summary_metrics(df_results: pd.DataFrame, 
                                    selected_region: str = None) -> None:
    """
    Render summary KPI cards for the selected region or all regions.
    """
    df_regional = aggregate_by_region(df_results)
    
    if selected_region and selected_region in df_regional['REGION'].values:
        region_data = df_regional[df_regional['REGION'] == selected_region].iloc[0]
        region_name = REGION_COORDS.get(selected_region, {}).get('full_name', selected_region)
    else:
        # All regions aggregate
        region_name = "Global"
        region_data = {
            'AVG_ROI': df_regional['AVG_ROI'].mean() if not df_regional.empty else 0,
            'TOTAL_SPEND': df_regional['TOTAL_SPEND'].sum() if not df_regional.empty else 0,
            'CHANNEL_COUNT': df_regional['CHANNEL_COUNT'].sum() if not df_regional.empty else 0,
            'ESTIMATED_REVENUE': df_regional['ESTIMATED_REVENUE'].sum() if not df_regional.empty else 0
        }
    
    if isinstance(region_data, pd.Series):
        region_data = region_data.to_dict()
    
    avg_roi = region_data.get('AVG_ROI', 0)
    total_spend = region_data.get('TOTAL_SPEND', 0)
    channel_count = region_data.get('CHANNEL_COUNT', 0)
    est_revenue = region_data.get('ESTIMATED_REVENUE', 0)
    
    cols = st.columns(4)
    
    with cols[0]:
        st.metric(
            f"{region_name} Avg ROI",
            f"{avg_roi:.2f}x",
            help="Average ROI across channels in this region"
        )
    
    with cols[1]:
        st.metric(
            "Total Spend",
            f"${total_spend/1e6:.1f}M",
            help="Sum of current spend across channels"
        )
    
    with cols[2]:
        # Format negative revenue properly
        rev_display = f"${est_revenue/1e6:.1f}M" if est_revenue >= 0 else f"-${abs(est_revenue)/1e6:.1f}M"
        st.metric(
            "Est. Revenue",
            rev_display,
            help="Estimated attributed revenue (spend × ROI). Negative indicates model uncertainty or measurement issues."
        )
    
    with cols[3]:
        st.metric(
            "Channels",
            f"{int(channel_count)}",
            help="Number of channel-region combinations"
        )

