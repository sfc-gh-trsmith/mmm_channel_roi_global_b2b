"""
Enforced visual styling for the MMM ROI Engine.
Ensures brand consistency regardless of user system theme.

LIGHT MODE THEME - Clean, professional appearance for demos.
"""
import streamlit as st

# Brand Color Palette
COLOR_PRIMARY = "#0068C9"      # Snowflake blue
COLOR_SECONDARY = "#11567F"    # Deep blue
COLOR_ACCENT = "#D95F02"       # Orange highlight
COLOR_SUCCESS = "#28A745"      # Green for positive metrics
COLOR_WARNING = "#F39C12"      # Amber for caution
COLOR_DANGER = "#DC3545"       # Red for negative metrics

# Background colors (Light Mode)
BG_LIGHT = "#FFFFFF"
BG_CARD = "#F8F9FA"
BG_HOVER = "#E9ECEF"
BG_SIDEBAR = "#F1F3F4"

# Text colors
TEXT_PRIMARY = "#1F2937"       # Dark gray for main text
TEXT_SECONDARY = "#6B7280"     # Medium gray for secondary text
TEXT_MUTED = "#9CA3AF"         # Light gray for muted text

# Typography
FONT_DISPLAY = "'Instrument Sans', 'DM Sans', sans-serif"
FONT_BODY = "'Inter', 'Source Sans Pro', sans-serif"
FONT_MONO = "'JetBrains Mono', 'Fira Code', monospace"


def inject_custom_css():
    """
    Inject brand-consistent CSS with LIGHT MODE theme.
    Call this at the top of every page.
    """
    st.markdown(f'''
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* ===== ROOT THEME OVERRIDE - LIGHT MODE ===== */
    .stApp {{
        background: linear-gradient(165deg, {BG_LIGHT} 0%, {BG_CARD} 100%);
    }}
    
    /* ===== TYPOGRAPHY ===== */
    html, body, [class*="css"] {{
        font-family: {FONT_BODY};
        color: {TEXT_PRIMARY};
    }}
    
    h1, h2, h3 {{
        font-family: {FONT_DISPLAY};
        font-weight: 700;
        letter-spacing: -0.02em;
        color: {TEXT_PRIMARY};
    }}
    
    h1 {{
        background: linear-gradient(135deg, {COLOR_PRIMARY} 0%, {COLOR_SECONDARY} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    
    code {{
        font-family: {FONT_MONO};
    }}
    
    /* ===== METRIC CARDS ===== */
    [data-testid="stMetric"] {{
        background: {BG_LIGHT};
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }}
    
    [data-testid="stMetricValue"] {{
        font-family: {FONT_DISPLAY};
        font-weight: 700;
        color: {TEXT_PRIMARY};
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {TEXT_SECONDARY};
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
        border: 1px solid #E5E7EB;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 8px;
        color: {TEXT_SECONDARY};
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        color: {COLOR_PRIMARY};
        background: rgba(0, 104, 201, 0.08);
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
        box-shadow: 0 2px 8px rgba(0, 104, 201, 0.25);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 104, 201, 0.35);
    }}
    
    /* ===== SEGMENTED CONTROL ===== */
    .segmented-control {{
        display: inline-flex;
        background: {BG_CARD};
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 4px;
        gap: 0;
    }}
    
    .segmented-control .seg-btn {{
        padding: 0.5rem 1rem;
        border: none;
        background: transparent;
        color: {TEXT_SECONDARY};
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
        background: rgba(0, 104, 201, 0.08);
    }}
    
    .segmented-control .seg-btn.active {{
        background: linear-gradient(135deg, {COLOR_PRIMARY} 0%, {COLOR_SECONDARY} 100%);
        color: white;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(0, 104, 201, 0.25);
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
        background: {BG_SIDEBAR} !important;
        border-right: 1px solid #E5E7EB;
    }}
    
    [data-testid="stSidebar"] [data-testid="stMarkdown"] {{
        color: {TEXT_PRIMARY} !important;
    }}
    
    /* Sidebar navigation links */
    [data-testid="stSidebar"] a {{
        color: {TEXT_PRIMARY} !important;
        text-decoration: none;
    }}
    
    [data-testid="stSidebar"] a:hover {{
        color: {COLOR_PRIMARY} !important;
    }}
    
    /* Sidebar navigation list items */
    [data-testid="stSidebar"] li {{
        color: {TEXT_PRIMARY} !important;
    }}
    
    [data-testid="stSidebar"] .stPageLink {{
        color: {TEXT_PRIMARY} !important;
    }}
    
    [data-testid="stSidebar"] .stPageLink:hover {{
        color: {COLOR_PRIMARY} !important;
        background: rgba(0, 104, 201, 0.08) !important;
    }}
    
    /* Sidebar page link text - Streamlit 1.x specific */
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] {{
        background: transparent !important;
    }}
    
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] span {{
        color: {TEXT_PRIMARY} !important;
    }}
    
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a {{
        color: {TEXT_PRIMARY} !important;
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        transition: all 0.2s ease;
    }}
    
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {{
        color: {COLOR_PRIMARY} !important;
        background: rgba(0, 104, 201, 0.08) !important;
    }}
    
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {{
        background: rgba(0, 104, 201, 0.12) !important;
        color: {COLOR_PRIMARY} !important;
        font-weight: 600;
        border-left: 3px solid {COLOR_PRIMARY};
    }}
    
    /* Force all sidebar text to be dark */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {{
        color: {TEXT_PRIMARY} !important;
    }}
    
    /* Sidebar title/header */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: {TEXT_PRIMARY} !important;
        -webkit-text-fill-color: {TEXT_PRIMARY} !important;
        background: none !important;
    }}
    
    /* Sidebar inner content area */
    [data-testid="stSidebarContent"] {{
        background: transparent !important;
    }}
    
    /* Sidebar collapse button */
    [data-testid="stSidebar"] button {{
        color: {TEXT_SECONDARY} !important;
    }}
    
    [data-testid="stSidebar"] button:hover {{
        color: {COLOR_PRIMARY} !important;
    }}
    
    /* ===== EXPANDERS (if used) ===== */
    .streamlit-expanderHeader {{
        background: {BG_CARD};
        border-radius: 8px;
        border: 1px solid #E5E7EB;
    }}
    
    /* ===== SLIDERS ===== */
    .stSlider > div > div > div {{
        background: linear-gradient(90deg, {COLOR_PRIMARY} 0%, {COLOR_SECONDARY} 100%);
    }}
    
    .stSlider > div > div > div > div {{
        background: {COLOR_PRIMARY};
        box-shadow: 0 0 8px rgba(0, 104, 201, 0.4);
    }}
    
    /* ===== SELECTBOX ===== */
    .stSelectbox > div > div {{
        background: {BG_LIGHT};
        border: 1px solid #E5E7EB;
        border-radius: 8px;
    }}
    
    /* ===== DATA FRAMES ===== */
    .stDataFrame {{
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #E5E7EB;
    }}
    
    /* ===== PLOTLY CHARTS ===== */
    .js-plotly-plot .plotly {{
        border-radius: 12px;
    }}
    
    /* ===== CHAT INPUT ===== */
    .stChatInput > div {{
        background: {BG_LIGHT};
        border: 1px solid #E5E7EB;
        border-radius: 12px;
    }}
    
    /* ===== PERSONA CARDS ===== */
    .persona-card {{
        background: {BG_LIGHT};
        border: 1px solid #E5E7EB;
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
        box-shadow: 0 8px 24px rgba(0, 104, 201, 0.15);
    }}
    
    .persona-title {{
        font-family: {FONT_DISPLAY};
        font-size: 1.4rem;
        font-weight: 700;
        color: {TEXT_PRIMARY};
        margin-bottom: 0.5rem;
    }}
    
    .persona-subtitle {{
        color: {COLOR_PRIMARY};
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }}
    
    .persona-description {{
        color: {TEXT_SECONDARY};
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
        background: rgba(0, 104, 201, 0.06);
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
        background: {BG_CARD};
        border: 1px solid #E5E7EB;
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
        color: {TEXT_PRIMARY};
    }}
    
    .quick-stat-label {{
        color: {TEXT_SECONDARY};
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
        background: {BG_CARD};
        border: 1px solid #E5E7EB;
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
        color: {TEXT_SECONDARY};
        font-size: 0.9rem;
        line-height: 1.6;
    }}
    
    .learn-more-content strong {{
        color: {TEXT_PRIMARY};
    }}
    
    /* ===== CONFIDENCE METRIC CARDS ===== */
    .confidence-metric {{
        background: {BG_LIGHT};
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        text-align: center;
    }}
    
    .confidence-value {{
        font-family: {FONT_DISPLAY};
        font-size: 1.8rem;
        font-weight: 700;
        color: {TEXT_PRIMARY};
    }}
    
    .confidence-range {{
        font-family: {FONT_MONO};
        font-size: 0.85rem;
        color: {TEXT_SECONDARY};
        margin-top: 0.25rem;
    }}
    
    .confidence-label {{
        color: {TEXT_SECONDARY};
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
        background: rgba(40, 167, 69, 0.12);
        color: {COLOR_SUCCESS};
        border: 1px solid rgba(40, 167, 69, 0.25);
    }}
    
    .significance-badge.uncertain {{
        background: rgba(243, 156, 18, 0.12);
        color: {COLOR_WARNING};
        border: 1px solid rgba(243, 156, 18, 0.25);
    }}
    
    .significance-badge.negative {{
        background: rgba(220, 53, 69, 0.12);
        color: {COLOR_DANGER};
        border: 1px solid rgba(220, 53, 69, 0.25);
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
        background: rgba(40, 167, 69, 0.12);
        color: {COLOR_SUCCESS};
    }}
    
    .zone-badge.diminishing {{
        background: rgba(243, 156, 18, 0.12);
        color: {COLOR_WARNING};
    }}
    
    .zone-badge.saturated {{
        background: rgba(220, 53, 69, 0.12);
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
        background: rgba(0, 104, 201, 0.08);
        border: 1px solid rgba(0, 104, 201, 0.2);
        border-radius: 6px;
        padding: 0.2rem 0.5rem;
        font-size: 0.75rem;
        color: {TEXT_SECONDARY};
    }}
    
    .param-pill .param-name {{
        color: {TEXT_MUTED};
    }}
    
    .param-pill .param-value {{
        font-family: {FONT_MONO};
        color: {COLOR_PRIMARY};
    }}
    
    /* ===== RECOMMENDATION CARD ===== */
    .recommendation-card {{
        background: rgba(0, 104, 201, 0.04);
        border: 1px solid {COLOR_PRIMARY};
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }}
    
    .recommendation-card.high-confidence {{
        border-color: {COLOR_SUCCESS};
        background: rgba(40, 167, 69, 0.04);
    }}
    
    .recommendation-card.uncertain {{
        border-color: {COLOR_WARNING};
        background: rgba(243, 156, 18, 0.04);
    }}
    
    .recommendation-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.75rem;
    }}
    
    .recommendation-title {{
        font-weight: 600;
        color: {TEXT_PRIMARY};
    }}
    
    /* ===== PRIORITY ACTION CARDS (LIGHT MODE) ===== */
    .priority-action-card {{
        background: rgba(0, 104, 201, 0.08);
        border: 1px solid rgba(0, 104, 201, 0.25);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }}
    
    .priority-action-card .badge {{
        font-size: 0.75rem;
        margin-bottom: 0.5rem;
    }}
    
    .priority-action-card .channel-name {{
        font-weight: 600;
        color: {TEXT_PRIMARY};
        margin-bottom: 0.25rem;
    }}
    
    .priority-action-card .metric-value {{
        font-size: 1.5rem;
        color: {COLOR_PRIMARY};
        font-weight: 700;
    }}
    
    .priority-action-card .metric-label {{
        font-size: 0.8rem;
        color: {TEXT_SECONDARY};
    }}
    
    .priority-action-card .spend-info {{
        font-size: 0.75rem;
        color: {TEXT_MUTED};
    }}
    
    .priority-action-card .action-text {{
        font-size: 0.9rem;
        color: {TEXT_SECONDARY};
        margin-top: 0.5rem;
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
    """Return a consistent Plotly template matching our brand (LIGHT MODE)."""
    return {
        'layout': {
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font': {
                'family': FONT_BODY,
                'color': TEXT_PRIMARY
            },
            'title': {
                'font': {
                    'family': FONT_DISPLAY,
                    'size': 18,
                    'color': TEXT_PRIMARY
                }
            },
            'xaxis': {
                'gridcolor': 'rgba(0,0,0,0.08)',
                'zerolinecolor': 'rgba(0,0,0,0.15)'
            },
            'yaxis': {
                'gridcolor': 'rgba(0,0,0,0.08)',
                'zerolinecolor': 'rgba(0,0,0,0.15)'
            },
            'colorway': [COLOR_PRIMARY, COLOR_ACCENT, COLOR_SUCCESS, COLOR_WARNING, '#9B59B6', '#1ABC9C']
        }
    }


def apply_plotly_theme(fig):
    """Apply our brand theme to a Plotly figure (LIGHT MODE)."""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family=FONT_BODY,
            color=TEXT_PRIMARY
        ),
        title_font=dict(
            family=FONT_DISPLAY,
            size=18,
            color=TEXT_PRIMARY
        ),
        xaxis=dict(
            gridcolor='rgba(0,0,0,0.08)',
            zerolinecolor='rgba(0,0,0,0.15)'
        ),
        yaxis=dict(
            gridcolor='rgba(0,0,0,0.08)',
            zerolinecolor='rgba(0,0,0,0.15)'
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color=TEXT_SECONDARY)
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
        <div style="color: {TEXT_SECONDARY};">{content}</div>
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
            color='rgba(0, 0, 0, 0.3)',
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
        fillcolor='rgba(0, 104, 201, 0.12)',
        line=dict(width=0),
        name=name,
        showlegend=True,
        hoverinfo='skip'
    ))
    return fig


def add_saturation_zone_annotation(fig, gamma: float, max_spend: float, 
                                    y_range: tuple = None, alpha: float = 2.5):
    """
    Add zone annotations based on position relative to gamma (half-saturation point).
    
    For the Hill function x^α / (x^α + γ^α):
    - At x = γ, response = 50% of maximum and slope is steepest
    - Before γ: accelerating returns (but response still growing fast)
    - After γ: diminishing returns
    - Well beyond γ: saturation (curve nearly flat)
    
    Zone logic:
    - High Efficiency (GREEN): 0 to 0.5*γ - strong growth region
    - Diminishing Returns (YELLOW): 0.5*γ to 1.5*γ - still valuable but slowing
    - Saturation (RED): beyond 1.5*γ - minimal incremental gain
    """
    diminishing_start = gamma * 0.5
    saturation_start = gamma * 1.5
    
    fig.add_vrect(
        x0=0,
        x1=diminishing_start,
        fillcolor="rgba(40, 167, 69, 0.12)",
        layer="below",
        line_width=0,
        annotation_text="High Efficiency",
        annotation_position="top left",
        annotation_font=dict(color=COLOR_SUCCESS, size=10)
    )
    
    if diminishing_start < saturation_start:
        fig.add_vrect(
            x0=diminishing_start,
            x1=min(saturation_start, max_spend),
            fillcolor="rgba(243, 156, 18, 0.10)",
            layer="below",
            line_width=0,
            annotation_text="Diminishing Returns",
            annotation_position="top left",
            annotation_font=dict(color=COLOR_WARNING, size=10)
        )
    
    if saturation_start < max_spend:
        fig.add_vrect(
            x0=saturation_start,
            x1=max_spend,
            fillcolor="rgba(220, 53, 69, 0.12)",
            layer="below",
            line_width=0,
            annotation_text="Saturation",
            annotation_position="top left",
            annotation_font=dict(color=COLOR_DANGER, size=10)
        )
    
    fig.add_vline(
        x=gamma,
        line=dict(color=COLOR_WARNING, width=2, dash='dash'),
        annotation_text="γ (50% Response)",
        annotation_position="top right",
        annotation_font=dict(color=COLOR_WARNING, size=10)
    )
    
    return fig


def format_ci_string(value: float, ci_lower: float, ci_upper: float, 
                     precision: int = 2, prefix: str = "", suffix: str = "x") -> str:
    """
    Format a value with its confidence interval as a string.
    
    Example: format_ci_string(2.3, 1.9, 2.7) -> "2.30x [1.90 - 2.70]"
    """
    return f"{prefix}{value:.{precision}f}{suffix} [{ci_lower:.{precision}f} - {ci_upper:.{precision}f}]"


def render_priority_action_card(channel: str, marginal_roi: float, spend: float,
                                 is_significant: bool = True) -> str:
    """
    Generate HTML for a priority action card (LIGHT MODE optimized).
    
    Use for displaying top recommended channels to invest in.
    """
    badge_text = "[High Confidence]" if is_significant else "[Uncertain]"
    badge_color = COLOR_SUCCESS if is_significant else COLOR_WARNING
    spend_label = f"${spend/1e6:.0f}M spend" if spend >= 1e6 else f"${spend/1e3:.0f}K spend"
    
    return f'''
    <div class="priority-action-card">
        <div class="badge" style="color: {badge_color};">{badge_text}</div>
        <div class="channel-name">{channel}</div>
        <div class="metric-value">{marginal_roi:.2f}x</div>
        <div class="metric-label">Marginal ROI</div>
        <div class="spend-info">{spend_label}</div>
        <div class="action-text">→ Increase investment</div>
    </div>
    '''


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
