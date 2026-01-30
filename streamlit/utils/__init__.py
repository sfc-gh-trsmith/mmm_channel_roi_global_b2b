"""
MMM ROI Engine Utilities

Shared utilities for data loading, styling, Cortex AI integration, and educational content.
"""

from utils.data_loader import run_queries_parallel
from utils.styling import (
    inject_custom_css,
    render_persona_card,
    render_story_section,
    render_insight_callout,
    render_quick_stats,
    apply_plotly_theme,
    get_plotly_template,
    # Enhanced components for descriptive/prescriptive MMM
    render_learn_more_panel,
    render_confidence_metric,
    render_significance_badge,
    render_zone_badge,
    render_adstock_decay_viz,
    render_parameter_pill,
    render_recommendation_card,
    add_error_bars_to_bar_chart,
    add_confidence_band_to_line_chart,
    add_saturation_zone_annotation,
    format_ci_string,
    calculate_confidence_level,
    # Colors
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_ACCENT,
    COLOR_SUCCESS,
    COLOR_WARNING,
    COLOR_DANGER,
    BG_LIGHT,
    BG_CARD,
    BG_HOVER
)
from utils.cortex_analyst import (
    send_analyst_message,
    parse_analyst_response,
    execute_analyst_sql,
    render_analyst_chat,
    render_example_queries,
    EXAMPLE_QUERIES
)
from utils.explanations import (
    EXPLANATIONS,
    TOOLTIPS,
    get_explanation,
    get_explanations_for_persona,
    get_explanations_by_depth,
    render_explanation_expander,
    get_tooltip
)

__all__ = [
    # Data loading
    'run_queries_parallel',
    
    # Styling - Core
    'inject_custom_css',
    'render_persona_card',
    'render_story_section',
    'render_insight_callout',
    'render_quick_stats',
    'apply_plotly_theme',
    'get_plotly_template',
    
    # Styling - Enhanced Components
    'render_learn_more_panel',
    'render_confidence_metric',
    'render_significance_badge',
    'render_zone_badge',
    'render_adstock_decay_viz',
    'render_parameter_pill',
    'render_recommendation_card',
    'add_error_bars_to_bar_chart',
    'add_confidence_band_to_line_chart',
    'add_saturation_zone_annotation',
    'format_ci_string',
    'calculate_confidence_level',
    
    # Colors
    'COLOR_PRIMARY',
    'COLOR_SECONDARY',
    'COLOR_ACCENT',
    'COLOR_SUCCESS',
    'COLOR_WARNING',
    'COLOR_DANGER',
    'BG_LIGHT',
    'BG_CARD',
    'BG_HOVER',
    
    # Cortex Analyst
    'send_analyst_message',
    'parse_analyst_response',
    'execute_analyst_sql',
    'render_analyst_chat',
    'render_example_queries',
    'EXAMPLE_QUERIES',
    
    # Explanations
    'EXPLANATIONS',
    'TOOLTIPS',
    'get_explanation',
    'get_explanations_for_persona',
    'get_explanations_by_depth',
    'render_explanation_expander',
    'get_tooltip'
]
