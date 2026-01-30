# Global B2B Marketing Mix Modeling (MMM) - AI Coding Guidelines

## Architecture Overview

This is a **Snowflake-native** Marketing Mix Modeling solution with three tiers:
1. **Data Layer** (`sql/`): Snowflake schemas (RAW → ATOMIC → DIMENSIONAL → MMM)
2. **ML Layer** (`notebooks/`): Snowpark ML training with Nevergrad optimization
3. **App Layer** (`streamlit/`): Streamlit in Snowflake with persona-based routing

**Key Insight**: Revenue attribution flows through `Revenue → Opportunity → Campaign → Channel`. The MMM learns adstock decay and Hill saturation parameters per channel.

## Data Schema Pattern

```
ATOMIC.MEDIA_SPEND_DAILY  ──┐
ATOMIC.OPPORTUNITY        ──┼── DIMENSIONAL.V_MMM_INPUT_WEEKLY ── MMM.MODEL_RESULTS
ATOMIC.ACTUAL_FINANCIAL_RESULT ─┘                                 MMM.RESPONSE_CURVES
```

- **Dimension tables** use SCD2 (`IS_CURRENT_FLAG`, `VALID_FROM/TO_TIMESTAMP`)
- **Hierarchy tables** are self-referential (Geography, Product_Category, Organization)
- **Views** flatten hierarchies for consumption (e.g., `DIM_GEOGRAPHY_HIERARCHY`)

## Developer Workflows

```bash
# Full deployment
./deploy.sh

# Targeted deployment
./deploy.sh --only-streamlit    # Streamlit only
./deploy.sh --only-data         # Data load only
./deploy.sh --only-sql          # SQL setup only

# Runtime operations
./run.sh main                   # Execute MMM training notebook
./run.sh streamlit              # Get Streamlit app URL
./run.sh status                 # Check resource status
```

## Streamlit Conventions

### Data Loading Pattern
Use centralized queries from `streamlit/utils/data_loader.py`:
```python
from utils.data_loader import run_queries_parallel, QUERIES

data = run_queries_parallel(session, {
    "CURVES": QUERIES["CURVES"],
    "RESULTS": QUERIES["RESULTS"],
})
```

### Page Structure
- Entry: `mmm_roi_app.py` (persona routing hub)
- Pages: `pages/1_Strategic_Dashboard.py`, `2_Simulator.py`, `3_Model_Explorer.py`
- Use `st.switch_page()` for navigation between pages

### Styling
Import from `utils/styling.py`:
```python
from utils.styling import (
    inject_custom_css, render_persona_card, COLOR_PRIMARY, apply_plotly_theme
)
```

### Light Mode Requirements (CRITICAL)
**This app is optimized for LIGHT MODE.** All custom HTML/CSS must be readable on light backgrounds.

**DO NOT USE:**
- `color: white` - invisible on light backgrounds
- `color: rgba(255,255,255,...)` - invisible/hard to read
- Hard-coded dark background colors in inline styles

**ALWAYS USE** the constants from `utils/styling.py`:
```python
# Text colors for light mode
TEXT_PRIMARY = "#1F2937"      # Main text (dark gray)
TEXT_SECONDARY = "#6B7280"    # Secondary text (medium gray)
TEXT_MUTED = "#9CA3AF"        # Muted/subtle text (light gray)

# Brand colors (these work on light backgrounds)
COLOR_PRIMARY = "#0068C9"     # Snowflake blue
COLOR_SUCCESS = "#28A745"     # Green
COLOR_WARNING = "#F39C12"     # Amber
COLOR_DANGER = "#DC3545"      # Red
```

**For custom card styles**, use the CSS classes defined in `styling.py`:
- `.priority-action-card` - For action/recommendation cards
- `.recommendation-card` - For detailed recommendations
- `.confidence-metric` - For metrics with confidence intervals
- `.persona-card` - For persona selection UI

**Example - WRONG vs RIGHT:**
```python
# WRONG - invisible text in light mode
st.markdown('<p style="color: rgba(255,255,255,0.6);">Text</p>', unsafe_allow_html=True)

# RIGHT - uses light-mode-safe colors
from utils.styling import TEXT_SECONDARY
st.markdown(f'<p style="color: {TEXT_SECONDARY};">Text</p>', unsafe_allow_html=True)
```

### Session State Pattern
For sliders with presets, update `st.session_state[f"slider_{channel}"]` BEFORE rendering, then `st.rerun()`:
```python
if optimize_clicked:
    for c in channels:
        st.session_state[f"slider_{c}"] = computed_value
    st.rerun()
```

## Model Parameters

MMM response curves use Hill saturation: `revenue = max_revenue * (spend^α / (spend^α + γ^α))`
- `α` (alpha): Shape parameter, controls curve steepness
- `γ` (gamma): Half-saturation point (spend where response = 50% max)
- `θ` (theta): Adstock decay rate (0-1, higher = longer carryover)

Efficiency zones based on gamma: `<0.5γ` = EFFICIENT, `0.5-1.5γ` = DIMINISHING, `>1.5γ` = SATURATED

## SQL Naming

- Tables: `ATOMIC.{TABLE_NAME}`, `MMM.{TABLE_NAME}`
- Views: `DIMENSIONAL.V_{VIEW_NAME}`, `MMM.V_{VIEW_NAME}`
- Stages: `DATA_STAGE`, `MODELS_STAGE`
- Use `ZEROIFNULL()` and `DIV0()` for safe aggregations

## Testing

Before committing Streamlit changes:
1. Deploy: `./deploy.sh --only-streamlit`
2. Verify app loads at URL from `./run.sh streamlit`
3. Test slider interactions and data flow

For SQL changes:
1. Run targeted SQL: `snow sql -q "SELECT * FROM MMM.V_ROI_BY_CHANNEL LIMIT 5"`
2. Verify views compile: Check `DIMENSIONAL.V_MMM_INPUT_WEEKLY` has data

## Common Gotchas

- **Channel naming**: Format is `{Platform}_{Region}_{Segment}` (e.g., `Facebook_NA_ALL`)
- **Session state warnings**: Don't pass `value=` when using `key=` with session state sliders
- **Snowflake session**: Always use `get_active_session()` from `snowflake.snowpark.context`
- **Response curves floor at 0**: Negative predictions indicate model uncertainty, not actual negative returns
