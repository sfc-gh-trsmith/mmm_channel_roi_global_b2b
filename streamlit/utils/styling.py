"""
Enforced visual styling for the MMM ROI Engine.
Ensures brand consistency regardless of user system theme.
"""
import streamlit as st

# Brand Color Palette
COLOR_PRIMARY = "#29B5E8"      # Cyan accent
COLOR_SECONDARY = "#11567F"    # Deep blue
COLOR_ACCENT = "#D95F02"       # Orange highlight
COLOR_SUCCESS = "#2ECC71"      # Green for positive metrics
COLOR_WARNING = "#F39C12"      # Amber for caution
COLOR_DANGER = "#E74C3C"       # Red for negative metrics

# Background colors
BG_DARK = "#0E1117"
BG_CARD = "#1a1f2e"
BG_HOVER = "#252d3d"

# Typography
FONT_DISPLAY = "'Instrument Sans', 'DM Sans', sans-serif"
FONT_BODY = "'Inter', 'Source Sans Pro', sans-serif"
FONT_MONO = "'JetBrains Mono', 'Fira Code', monospace"


def inject_custom_css():
    """
    Inject brand-consistent CSS regardless of user system theme.
    Call this at the top of every page.
    """
    st.markdown(f'''
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* ===== ROOT THEME OVERRIDE ===== */
    .stApp {{
        background: linear-gradient(165deg, {BG_DARK} 0%, {BG_CARD} 50%, #0d1520 100%);
    }}
    
    /* ===== TYPOGRAPHY ===== */
    html, body, [class*="css"] {{
        font-family: {FONT_BODY};
    }}
    
    h1, h2, h3 {{
        font-family: {FONT_DISPLAY};
        font-weight: 700;
        letter-spacing: -0.02em;
    }}
    
    h1 {{
        background: linear-gradient(135deg, {COLOR_PRIMARY} 0%, #7DD3FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    
    code {{
        font-family: {FONT_MONO};
    }}
    
    /* ===== METRIC CARDS ===== */
    [data-testid="stMetric"] {{
        background: linear-gradient(145deg, {BG_CARD} 0%, {BG_HOVER} 100%);
        border: 1px solid rgba(41, 181, 232, 0.2);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }}
    
    [data-testid="stMetricValue"] {{
        font-family: {FONT_DISPLAY};
        font-weight: 700;
        color: #ffffff;
    }}
    
    [data-testid="stMetricLabel"] {{
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    
    [data-testid="stMetricDelta"] svg {{
        display: none;
    }}
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: {BG_CARD};
        padding: 0.5rem;
        border-radius: 12px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 8px;
        color: rgba(255, 255, 255, 0.6);
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        color: {COLOR_PRIMARY};
        background: rgba(41, 181, 232, 0.1);
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {COLOR_PRIMARY} 0%, {COLOR_SECONDARY} 100%);
        color: white !important;
    }}
    
    /* ===== BUTTONS ===== */
    .stButton > button {{
        background: linear-gradient(135deg, {COLOR_PRIMARY} 0%, {COLOR_SECONDARY} 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(41, 181, 232, 0.3);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(41, 181, 232, 0.4);
    }}
    
    /* ===== SEGMENTED CONTROL ===== */
    .segmented-control {{
        display: inline-flex;
        background: {BG_CARD};
        border: 1px solid rgba(41, 181, 232, 0.2);
        border-radius: 12px;
        padding: 4px;
        gap: 0;
    }}
    
    .segmented-control .seg-btn {{
        padding: 0.5rem 1rem;
        border: none;
        background: transparent;
        color: rgba(255, 255, 255, 0.6);
        font-family: {FONT_BODY};
        font-size: 0.9rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        border-radius: 8px;
        white-space: nowrap;
    }}
    
    .segmented-control .seg-btn:hover {{
        color: {COLOR_PRIMARY};
        background: rgba(41, 181, 232, 0.1);
    }}
    
    .segmented-control .seg-btn.active {{
        background: linear-gradient(135deg, {COLOR_PRIMARY} 0%, {COLOR_SECONDARY} 100%);
        color: white;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(41, 181, 232, 0.3);
    }}
    
    .segmented-control .seg-btn .roi-badge {{
        font-family: {FONT_MONO};
        font-size: 0.8rem;
        opacity: 0.8;
        margin-left: 0.25rem;
    }}
    
    .segmented-control .seg-btn.active .roi-badge {{
        opacity: 1;
    }}
    
    /* Hide the actual radio buttons when using segmented control */
    .segmented-radio-hidden {{
        position: absolute;
        opacity: 0;
        pointer-events: none;
        height: 0;
        overflow: hidden;
    }}
    
    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {BG_CARD} 0%, {BG_DARK} 100%) !important;
        border-right: 1px solid rgba(41, 181, 232, 0.15);
    }}
    
    [data-testid="stSidebar"] [data-testid="stMarkdown"] {{
        color: rgba(255, 255, 255, 0.85) !important;
    }}
    
    /* Sidebar navigation links */
    [data-testid="stSidebar"] a {{
        color: rgba(255, 255, 255, 0.85) !important;
        text-decoration: none;
    }}
    
    [data-testid="stSidebar"] a:hover {{
        color: {COLOR_PRIMARY} !important;
    }}
    
    /* Sidebar navigation list items */
    [data-testid="stSidebar"] li {{
        color: rgba(255, 255, 255, 0.85) !important;
    }}
    
    [data-testid="stSidebar"] .stPageLink {{
        color: rgba(255, 255, 255, 0.85) !important;
    }}
    
    [data-testid="stSidebar"] .stPageLink:hover {{
        color: {COLOR_PRIMARY} !important;
        background: rgba(41, 181, 232, 0.1) !important;
    }}
    
    /* Sidebar page link text - Streamlit 1.x specific */
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] {{
        background: transparent !important;
    }}
    
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] span {{
        color: rgba(255, 255, 255, 0.85) !important;
    }}
    
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a {{
        color: rgba(255, 255, 255, 0.85) !important;
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        transition: all 0.2s ease;
    }}
    
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {{
        color: {COLOR_PRIMARY} !important;
        background: rgba(41, 181, 232, 0.15) !important;
    }}
    
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {{
        background: linear-gradient(135deg, rgba(41, 181, 232, 0.2) 0%, rgba(17, 86, 127, 0.2) 100%) !important;
        color: {COLOR_PRIMARY} !important;
        font-weight: 600;
        border-left: 3px solid {COLOR_PRIMARY};
    }}
    
    /* Force all sidebar text to be light */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {{
        color: rgba(255, 255, 255, 0.85) !important;
    }}
    
    /* Sidebar title/header */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: white !important;
        -webkit-text-fill-color: white !important;
        background: none !important;
    }}
    
    /* Sidebar inner content area */
    [data-testid="stSidebarContent"] {{
        background: transparent !important;
    }}
    
    /* Sidebar collapse button */
    [data-testid="stSidebar"] button {{
        color: rgba(255, 255, 255, 0.7) !important;
    }}
    
    [data-testid="stSidebar"] button:hover {{
        color: {COLOR_PRIMARY} !important;
    }}
    
    /* ===== EXPANDERS (if used) ===== */
    .streamlit-expanderHeader {{
        background: {BG_CARD};
        border-radius: 8px;
        border: 1px solid rgba(41, 181, 232, 0.2);
    }}
    
    /* ===== SLIDERS ===== */
    .stSlider > div > div > div {{
        background: linear-gradient(90deg, {COLOR_PRIMARY} 0%, {COLOR_SECONDARY} 100%);
    }}
    
    .stSlider > div > div > div > div {{
        background: {COLOR_PRIMARY};
        box-shadow: 0 0 10px rgba(41, 181, 232, 0.5);
    }}
    
    /* ===== SELECTBOX ===== */
    .stSelectbox > div > div {{
        background: {BG_CARD};
        border: 1px solid rgba(41, 181, 232, 0.3);
        border-radius: 8px;
    }}
    
    /* ===== DATA FRAMES ===== */
    .stDataFrame {{
        border-radius: 12px;
        overflow: hidden;
    }}
    
    /* ===== PLOTLY CHARTS ===== */
    .js-plotly-plot .plotly {{
        border-radius: 12px;
    }}
    
    /* ===== CHAT INPUT ===== */
    .stChatInput > div {{
        background: {BG_CARD};
        border: 1px solid rgba(41, 181, 232, 0.3);
        border-radius: 12px;
    }}
    
    /* ===== PERSONA CARDS ===== */
    .persona-card {{
        background: linear-gradient(145deg, {BG_CARD} 0%, {BG_HOVER} 100%);
        border: 1px solid rgba(41, 181, 232, 0.2);
        border-top: 4px solid {COLOR_PRIMARY};
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
        height: 100%;
    }}
    
    .persona-card:hover {{
        border-color: {COLOR_PRIMARY};
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(41, 181, 232, 0.2);
    }}
    
    .persona-title {{
        font-family: {FONT_DISPLAY};
        font-size: 1.4rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
    }}
    
    .persona-subtitle {{
        color: {COLOR_PRIMARY};
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }}
    
    .persona-description {{
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.9rem;
        line-height: 1.5;
    }}
    
    /* ===== STORY SECTIONS ===== */
    .story-section {{
        background: {BG_CARD};
        border-left: 4px solid {COLOR_PRIMARY};
        border-radius: 0 12px 12px 0;
        padding: 1.5rem;
        margin: 1rem 0;
    }}
    
    .story-label {{
        color: {COLOR_PRIMARY};
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }}
    
    /* ===== INSIGHT CALLOUT ===== */
    .insight-callout {{
        background: linear-gradient(135deg, rgba(41, 181, 232, 0.15) 0%, rgba(17, 86, 127, 0.15) 100%);
        border: 1px solid {COLOR_PRIMARY};
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }}
    
    .insight-callout h4 {{
        color: {COLOR_PRIMARY};
        margin-bottom: 0.5rem;
    }}
    
    /* ===== QUICK STATS BAR ===== */
    .quick-stats {{
        display: flex;
        justify-content: center;
        gap: 3rem;
        padding: 1rem;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 12px;
        margin-top: 2rem;
    }}
    
    .quick-stat {{
        text-align: center;
    }}
    
    .quick-stat-value {{
        font-family: {FONT_DISPLAY};
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
    }}
    
    .quick-stat-label {{
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    
    /* ===== HIDE STREAMLIT BRANDING ===== */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* ===== LEARN MORE PANELS ===== */
    .learn-more-panel {{
        background: linear-gradient(145deg, rgba(26, 31, 46, 0.8) 0%, rgba(37, 45, 61, 0.6) 100%);
        border: 1px solid rgba(41, 181, 232, 0.15);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
    }}
    
    .learn-more-header {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: {COLOR_PRIMARY};
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.75rem;
    }}
    
    .learn-more-content {{
        color: rgba(255, 255, 255, 0.75);
        font-size: 0.9rem;
        line-height: 1.6;
    }}
    
    .learn-more-content strong {{
        color: white;
    }}
    
    /* ===== CONFIDENCE METRIC CARDS ===== */
    .confidence-metric {{
        background: linear-gradient(145deg, {BG_CARD} 0%, {BG_HOVER} 100%);
        border: 1px solid rgba(41, 181, 232, 0.2);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        text-align: center;
    }}
    
    .confidence-value {{
        font-family: {FONT_DISPLAY};
        font-size: 1.8rem;
        font-weight: 700;
        color: white;
    }}
    
    .confidence-range {{
        font-family: {FONT_MONO};
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.5);
        margin-top: 0.25rem;
    }}
    
    .confidence-label {{
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.5rem;
    }}
    
    /* ===== SIGNIFICANCE BADGES ===== */
    .significance-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }}
    
    .significance-badge.significant {{
        background: rgba(46, 204, 113, 0.2);
        color: {COLOR_SUCCESS};
        border: 1px solid rgba(46, 204, 113, 0.3);
    }}
    
    .significance-badge.uncertain {{
        background: rgba(243, 156, 18, 0.2);
        color: {COLOR_WARNING};
        border: 1px solid rgba(243, 156, 18, 0.3);
    }}
    
    .significance-badge.negative {{
        background: rgba(231, 76, 60, 0.2);
        color: {COLOR_DANGER};
        border: 1px solid rgba(231, 76, 60, 0.3);
    }}
    
    /* ===== EFFICIENCY ZONES ===== */
    .zone-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
    }}
    
    .zone-badge.efficient {{
        background: rgba(46, 204, 113, 0.15);
        color: {COLOR_SUCCESS};
    }}
    
    .zone-badge.diminishing {{
        background: rgba(243, 156, 18, 0.15);
        color: {COLOR_WARNING};
    }}
    
    .zone-badge.saturated {{
        background: rgba(231, 76, 60, 0.15);
        color: {COLOR_DANGER};
    }}
    
    /* ===== ADSTOCK DECAY SPARKLINE ===== */
    .adstock-viz {{
        display: flex;
        align-items: flex-end;
        gap: 2px;
        height: 24px;
    }}
    
    .adstock-bar {{
        width: 4px;
        background: {COLOR_PRIMARY};
        border-radius: 2px;
        transition: height 0.2s;
    }}
    
    /* ===== PARAMETER PILL ===== */
    .param-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: rgba(41, 181, 232, 0.1);
        border: 1px solid rgba(41, 181, 232, 0.2);
        border-radius: 6px;
        padding: 0.2rem 0.5rem;
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.7);
    }}
    
    .param-pill .param-name {{
        color: rgba(255, 255, 255, 0.5);
    }}
    
    .param-pill .param-value {{
        font-family: {FONT_MONO};
        color: {COLOR_PRIMARY};
    }}
    
    /* ===== RECOMMENDATION CARD ===== */
    .recommendation-card {{
        background: linear-gradient(135deg, rgba(41, 181, 232, 0.1) 0%, rgba(17, 86, 127, 0.1) 100%);
        border: 1px solid {COLOR_PRIMARY};
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }}
    
    .recommendation-card.high-confidence {{
        border-color: {COLOR_SUCCESS};
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.1) 0%, rgba(46, 204, 113, 0.05) 100%);
    }}
    
    .recommendation-card.uncertain {{
        border-color: {COLOR_WARNING};
        background: linear-gradient(135deg, rgba(243, 156, 18, 0.1) 0%, rgba(243, 156, 18, 0.05) 100%);
    }}
    
    .recommendation-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.75rem;
    }}
    
    .recommendation-title {{
        font-weight: 600;
        color: white;
    }}
    </style>
    ''', unsafe_allow_html=True)


def render_persona_card(title: str, subtitle: str, description: str) -> str:
    """Generate HTML for a persona selection card."""
    return f'''
    <div class="persona-card">
        <div class="persona-title">{title}</div>
        <div class="persona-subtitle">{subtitle}</div>
        <div class="persona-description">{description}</div>
    </div>
    '''


def render_story_section(label: str, content: str) -> str:
    """Generate HTML for a story-driven section."""
    return f'''
    <div class="story-section">
        <div class="story-label">{label}</div>
        <div>{content}</div>
    </div>
    '''


def render_insight_callout(title: str, content: str) -> str:
    """Generate HTML for an insight/recommendation callout."""
    return f'''
    <div class="insight-callout">
        <h4>{title}</h4>
        <p>{content}</p>
    </div>
    '''


def render_quick_stats(stats: list) -> str:
    """
    Generate HTML for quick stats bar.
    stats: list of dicts with 'value' and 'label' keys
    """
    stats_html = ''.join([
        f'''<div class="quick-stat">
            <div class="quick-stat-value">{s['value']}</div>
            <div class="quick-stat-label">{s['label']}</div>
        </div>'''
        for s in stats
    ])
    return f'<div class="quick-stats">{stats_html}</div>'


def get_plotly_template():
    """Return a consistent Plotly template matching our brand."""
    return {
        'layout': {
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font': {
                'family': FONT_BODY,
                'color': 'rgba(255,255,255,0.85)'
            },
            'title': {
                'font': {
                    'family': FONT_DISPLAY,
                    'size': 18,
                    'color': 'white'
                }
            },
            'xaxis': {
                'gridcolor': 'rgba(255,255,255,0.1)',
                'zerolinecolor': 'rgba(255,255,255,0.2)'
            },
            'yaxis': {
                'gridcolor': 'rgba(255,255,255,0.1)',
                'zerolinecolor': 'rgba(255,255,255,0.2)'
            },
            'colorway': [COLOR_PRIMARY, COLOR_ACCENT, COLOR_SUCCESS, COLOR_WARNING, '#9B59B6', '#1ABC9C']
        }
    }


def apply_plotly_theme(fig):
    """Apply our brand theme to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family=FONT_BODY,
            color='rgba(255,255,255,0.85)'
        ),
        title_font=dict(
            family=FONT_DISPLAY,
            size=18,
            color='white'
        ),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(255,255,255,0.7)')
        )
    )
    return fig


# =============================================================================
# ENHANCED COMPONENTS FOR DESCRIPTIVE/PRESCRIPTIVE MMM
# =============================================================================

def render_learn_more_panel(title: str, content: str) -> str:
    """
    Generate HTML for a collapsible "Learn More" educational panel.
    
    Use this to add contextual explanations without cluttering the UI.
    Content can include HTML formatting (strong, em, ul/li).
    """
    return f'''
    <div class="learn-more-panel">
        <div class="learn-more-header">{title}</div>
        <div class="learn-more-content">{content}</div>
    </div>
    '''


def render_confidence_metric(value: str, ci_lower: float, ci_upper: float, 
                             label: str, is_significant: bool = True) -> str:
    """
    Generate HTML for a metric card with confidence interval range.
    
    Example: "2.3x [1.9 - 2.7]" for ROI with 90% CI.
    Shows significance badge if metric is statistically significant.
    """
    badge = render_significance_badge(is_significant)
    return f'''
    <div class="confidence-metric">
        <div class="confidence-value">{value}</div>
        <div class="confidence-range">[{ci_lower:.2f} - {ci_upper:.2f}]</div>
        <div class="confidence-label">{label}</div>
        {badge}
    </div>
    '''


def render_significance_badge(is_significant: bool, roi: float = None) -> str:
    """
    Generate HTML for a significance indicator badge.
    
    - Green: Statistically significant (CI excludes breakeven)
    - Amber: Uncertain (wide CI or crosses breakeven)
    - Red: Significant but negative (ROI < 1)
    """
    if is_significant:
        if roi is not None and roi < 1.0:
            return '<span class="significance-badge negative">Below Breakeven</span>'
        return '<span class="significance-badge significant">Significant</span>'
    else:
        return '<span class="significance-badge uncertain">Uncertain</span>'


def render_zone_badge(zone: str) -> str:
    """
    Generate HTML for an efficiency zone badge.
    
    Zones:
    - EFFICIENT (green): Marginal ROI > 1.5x
    - DIMINISHING (amber): Marginal ROI 0.8-1.5x
    - SATURATED (red): Marginal ROI < 0.8x
    """
    zone = zone.upper() if zone else 'UNKNOWN'
    if zone == 'EFFICIENT':
        return '<span class="zone-badge efficient">Efficient</span>'
    elif zone == 'DIMINISHING':
        return '<span class="zone-badge diminishing">Diminishing</span>'
    elif zone == 'SATURATED':
        return '<span class="zone-badge saturated">Saturated</span>'
    else:
        return '<span class="zone-badge">Unknown</span>'


def render_adstock_decay_viz(theta: float, weeks: int = 8) -> str:
    """
    Generate HTML for a mini sparkline showing adstock decay over time.
    
    Visual representation of how advertising effect decays week-over-week.
    Higher theta = slower decay = longer carryover effect.
    """
    heights = []
    value = 100
    for _ in range(weeks):
        heights.append(value)
        value *= theta
    
    bars_html = ''.join([
        f'<div class="adstock-bar" style="height: {h * 0.24}px;"></div>'
        for h in heights
    ])
    
    return f'<div class="adstock-viz">{bars_html}</div>'


def render_parameter_pill(name: str, value: float, format_str: str = ".2f") -> str:
    """
    Generate HTML for a compact parameter display pill.
    
    Used to show learned MMM parameters like decay rate, saturation point.
    """
    formatted_value = f"{value:{format_str}}"
    return f'<span class="param-pill"><span class="param-name">{name}:</span><span class="param-value">{formatted_value}</span></span>'


def render_recommendation_card(title: str, content: str, confidence: str = "medium",
                               predicted_lift: str = None) -> str:
    """
    Generate HTML for a styled recommendation card with confidence indicator.
    
    confidence: "high", "medium", or "low"
    """
    confidence_class = "high-confidence" if confidence == "high" else "uncertain" if confidence == "low" else ""
    
    lift_html = ""
    if predicted_lift:
        lift_html = f'<div style="margin-top: 0.75rem; color: {COLOR_SUCCESS}; font-weight: 600;">Predicted Lift: {predicted_lift}</div>'
    
    confidence_badge = {
        "high": f'<span class="significance-badge significant">High Confidence</span>',
        "medium": f'<span class="significance-badge uncertain">Medium Confidence</span>',
        "low": f'<span class="significance-badge negative">Low Confidence</span>'
    }.get(confidence, "")
    
    return f'''
    <div class="recommendation-card {confidence_class}">
        <div class="recommendation-header">
            <span class="recommendation-title">{title}</span>
            {confidence_badge}
        </div>
        <div style="color: rgba(255,255,255,0.8);">{content}</div>
        {lift_html}
    </div>
    '''


def add_error_bars_to_bar_chart(fig, ci_lower_values: list, ci_upper_values: list, 
                                 y_values: list, trace_index: int = 0):
    """
    Add horizontal error bars to a bar chart for confidence intervals.
    
    Updates the figure in place. Use for ROI charts with uncertainty.
    """
    import numpy as np
    
    error_plus = [u - y for u, y in zip(ci_upper_values, y_values)]
    error_minus = [y - l for y, l in zip(y_values, ci_lower_values)]
    
    fig.data[trace_index].update(
        error_x=dict(
            type='data',
            array=error_plus,
            arrayminus=error_minus,
            color='rgba(255, 255, 255, 0.4)',
            thickness=1.5,
            width=4
        )
    )
    return fig


def add_confidence_band_to_line_chart(fig, x_values: list, y_lower: list, 
                                       y_upper: list, name: str = "90% CI"):
    """
    Add a shaded confidence band to a line chart.
    
    Creates a filled area between lower and upper bounds.
    """
    import plotly.graph_objects as go
    
    # Reverse for proper fill
    x_band = list(x_values) + list(x_values)[::-1]
    y_band = list(y_upper) + list(y_lower)[::-1]
    
    fig.add_trace(go.Scatter(
        x=x_band,
        y=y_band,
        fill='toself',
        fillcolor='rgba(41, 181, 232, 0.15)',
        line=dict(width=0),
        name=name,
        showlegend=True,
        hoverinfo='skip'
    ))
    return fig


def add_saturation_zone_annotation(fig, gamma: float, max_spend: float, 
                                    y_range: tuple = None):
    """
    Add vertical line and shaded zone annotation for saturation point.
    
    gamma: Half-saturation spend level (from Hill function)
    Shows where diminishing returns become significant.
    """
    # Add vertical line at gamma (half-saturation)
    fig.add_vline(
        x=gamma,
        line=dict(color=COLOR_WARNING, width=2, dash='dash'),
        annotation_text="Half-Saturation",
        annotation_position="top right",
        annotation_font=dict(color=COLOR_WARNING, size=10)
    )
    
    # Add shaded "saturation zone" from 2*gamma onwards
    saturation_start = gamma * 2
    if saturation_start < max_spend:
        fig.add_vrect(
            x0=saturation_start,
            x1=max_spend,
            fillcolor="rgba(231, 76, 60, 0.1)",
            layer="below",
            line_width=0,
            annotation_text="Saturation Zone",
            annotation_position="top left",
            annotation_font=dict(color=COLOR_DANGER, size=9)
        )
    
    # Add "efficient zone" shading from 0 to gamma
    fig.add_vrect(
        x0=0,
        x1=gamma,
        fillcolor="rgba(46, 204, 113, 0.08)",
        layer="below",
        line_width=0,
        annotation_text="Efficient Zone",
        annotation_position="top left",
        annotation_font=dict(color=COLOR_SUCCESS, size=9)
    )
    
    return fig


def format_ci_string(value: float, ci_lower: float, ci_upper: float, 
                     precision: int = 2, prefix: str = "", suffix: str = "x") -> str:
    """
    Format a value with its confidence interval as a string.
    
    Example: format_ci_string(2.3, 1.9, 2.7) -> "2.30x [1.90 - 2.70]"
    """
    return f"{prefix}{value:.{precision}f}{suffix} [{ci_lower:.{precision}f} - {ci_upper:.{precision}f}]"


def calculate_confidence_level(ci_lower: float, ci_upper: float, 
                                value: float) -> str:
    """
    Determine confidence level based on CI width relative to estimate.
    
    Returns: "high", "medium", or "low"
    """
    ci_width = ci_upper - ci_lower
    relative_width = ci_width / abs(value) if value != 0 else float('inf')
    
    if relative_width < 0.3:
        return "high"
    elif relative_width < 0.6:
        return "medium"
    else:
        return "low"

