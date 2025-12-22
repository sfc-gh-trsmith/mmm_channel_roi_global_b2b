"""
Centralized Educational Content for MMM ROI Engine

This module contains all explanatory text, tooltips, and educational panels
used across the Streamlit application. Centralizing this content ensures:
1. Consistency across pages
2. Easy updates without touching UI code
3. Persona-appropriate messaging

Each explanation includes:
- title: Header for the learn-more panel
- icon: Emoji for visual context
- content: HTML-formatted explanation (can use <strong>, <em>, <ul>/<li>)
- persona: List of personas this content is relevant for
- depth: "overview" (CMO), "tactical" (Regional Lead), or "technical" (Data Scientist)
"""

# =============================================================================
# CORE MMM CONCEPTS
# =============================================================================

EXPLANATIONS = {
    # -------------------------------------------------------------------------
    # ROI & Attribution
    # -------------------------------------------------------------------------
    "roi_attribution": {
        "title": "What is ROI Attribution?",
        "content": """
            <strong>Marketing Mix Modeling (MMM)</strong> uses econometric techniques to measure 
            the true impact of each marketing channel on revenue—accounting for factors that 
            traditional attribution misses.
            <br><br>
            <strong>Why MMM beats last-click attribution:</strong>
            <ul>
                <li>Captures <strong>offline impact</strong> (TV, print, events)</li>
                <li>Accounts for <strong>time delays</strong> (B2B sales cycles of 6-18 months)</li>
                <li>Separates <strong>marketing lift</strong> from baseline demand and seasonality</li>
                <li>Handles <strong>distributor sales</strong> where direct attribution is impossible</li>
            </ul>
            The ROI shown here represents incremental revenue generated per dollar spent,
            isolated from other factors.
        """,
        "persona": ["cmo", "regional_lead"],
        "depth": "overview"
    },
    
    "marginal_vs_average_roi": {
        "title": "Average ROI vs. Marginal ROI",
        "content": """
            <strong>Average ROI</strong> = Total revenue attributed / Total spend
            <br>
            "What did we get back per dollar historically?"
            <br><br>
            <strong>Marginal ROI</strong> = Additional revenue from the NEXT dollar spent
            <br>
            "What will we get back if we spend more NOW?"
            <br><br>
            <strong>Why marginal matters more:</strong>
            <ul>
                <li>A channel with 5x average ROI might be <em>saturated</em> (0.5x marginal)</li>
                <li>A channel with 2x average ROI might have <em>headroom</em> (3x marginal)</li>
                <li>Budget decisions should be based on the <em>next dollar</em>, not past performance</li>
            </ul>
            The simulator uses marginal ROI to predict the actual impact of budget shifts.
        """,
        "persona": ["cmo", "regional_lead", "data_scientist"],
        "depth": "tactical"
    },
    
    # -------------------------------------------------------------------------
    # Confidence Intervals
    # -------------------------------------------------------------------------
    "confidence_intervals": {
        "title": "Understanding Confidence Intervals",
        "content": """
            A <strong>90% confidence interval</strong> means: if we repeated this analysis 
            many times with different data samples, 90% of the resulting intervals would 
            contain the true ROI value.
            <br><br>
            <strong>How to interpret:</strong>
            <ul>
                <li><strong>Narrow CI (e.g., 2.8-3.2)</strong>: High confidence—estimate is reliable</li>
                <li><strong>Wide CI (e.g., 1.5-4.5)</strong>: Uncertain—more data needed</li>
                <li><strong>CI excludes 1.0</strong>: Statistically significant positive ROI</li>
                <li><strong>CI includes 1.0</strong>: Cannot confidently say channel is profitable</li>
            </ul>
            Use this to prioritize decisions: <em>act confidently on narrow CIs, 
            investigate further on wide CIs</em>.
        """,
        "persona": ["cmo", "data_scientist"],
        "depth": "overview"
    },
    
    "statistical_significance": {
        "title": "What Does 'Significant' Mean?",
        "content": """
            <strong>Statistically significant</strong> means we can be confident the observed 
            effect is real, not just random noise in the data.
            <br><br>
            <strong>In this model:</strong>
            <ul>
                <li><strong>Significant</strong>: The 90% CI does NOT include breakeven (1.0x ROI)</li>
                <li><strong>Uncertain</strong>: The CI crosses 1.0—could be profitable or not</li>
            </ul>
            Significant channels have enough data and clear enough signal to trust the ROI estimate.
            Uncertain channels may need more investment to generate clearer signal, or may 
            genuinely have minimal impact.
        """,
        "persona": ["cmo", "regional_lead"],
        "depth": "overview"
    },
    
    # -------------------------------------------------------------------------
    # Adstock & Carryover
    # -------------------------------------------------------------------------
    "adstock_carryover": {
        "title": "How Adstock (Carryover) Works",
        "content": """
            <strong>Adstock</strong> captures the "memory" of advertising: today's ad spend 
            continues to influence buyers for weeks after exposure.
            <br><br>
            <strong>The decay rate (θ)</strong> determines how quickly the effect fades:
            <ul>
                <li><strong>θ = 0.8</strong> (Long carryover): Effect persists 4-6 weeks. 
                    Typical for LinkedIn B2B, brand awareness campaigns.</li>
                <li><strong>θ = 0.3</strong> (Short carryover): Effect fades in 1-2 weeks. 
                    Typical for paid search, direct response.</li>
            </ul>
            <strong>Why it matters:</strong> A channel with high carryover keeps working 
            even after you reduce spend—so you can afford to "pulse" budget rather 
            than maintain constant pressure.
        """,
        "persona": ["regional_lead", "data_scientist"],
        "depth": "tactical"
    },
    
    "adstock_technical": {
        "title": "Geometric Adstock Transformation",
        "content": """
            The model applies <strong>geometric adstock</strong> transformation:
            <br><br>
            <code>x_adstock[t] = x[t] + θ × x_adstock[t-1]</code>
            <br><br>
            This is equivalent to an infinite geometric series where each week's 
            effective spend includes decayed contributions from all prior weeks.
            <br><br>
            <strong>Half-life calculation:</strong> ln(0.5) / ln(θ) weeks
            <br>
            Example: θ = 0.7 → half-life ≈ 2 weeks
            <br><br>
            The <strong>θ parameter is learned per channel</strong> via Nevergrad 
            optimization, finding the decay rate that best explains the 
            spend-to-revenue relationship.
        """,
        "persona": ["data_scientist"],
        "depth": "technical"
    },
    
    # -------------------------------------------------------------------------
    # Saturation & Diminishing Returns
    # -------------------------------------------------------------------------
    "saturation_curves": {
        "title": "Reading Saturation Curves",
        "content": """
            <strong>Saturation curves</strong> show how each additional dollar of spend 
            generates progressively less incremental revenue—the law of diminishing returns.
            <br><br>
            <strong>Three zones on every curve:</strong>
            <ul>
                <li><strong style="color: #2ECC71;">Efficient Zone</strong> (steep slope): 
                    Each $1 returns $1.50+. Room to invest more.</li>
                <li><strong style="color: #F39C12;">Diminishing Zone</strong> (flattening): 
                    Each $1 returns $0.80-$1.50. Optimize carefully.</li>
                <li><strong style="color: #E74C3C;">Saturated Zone</strong> (flat): 
                    Each $1 returns &lt;$0.80. Shift budget elsewhere.</li>
            </ul>
            The <strong>half-saturation point (γ)</strong> is where you're at 50% of 
            maximum possible response—a key inflection point.
        """,
        "persona": ["cmo", "regional_lead"],
        "depth": "tactical"
    },
    
    "hill_function": {
        "title": "Hill Saturation Function",
        "content": """
            The model uses a <strong>Hill function</strong> (from pharmacology) to capture 
            diminishing returns:
            <br><br>
            <code>response = x^α / (γ^α + x^α)</code>
            <br><br>
            <strong>Parameters:</strong>
            <ul>
                <li><strong>α (alpha)</strong>: Shape/steepness. Higher = sharper S-curve.</li>
                <li><strong>γ (gamma)</strong>: Half-saturation point. Spend level where 
                    response = 50% of maximum.</li>
            </ul>
            <strong>Why Hill function?</strong>
            <ul>
                <li>Bounded output [0, 1] prevents extrapolation errors</li>
                <li>Interpretable γ parameter (easy to explain to stakeholders)</li>
                <li>Industry standard: used by Meta's Robyn, Google's LightweightMMM</li>
            </ul>
        """,
        "persona": ["data_scientist"],
        "depth": "technical"
    },
    
    # -------------------------------------------------------------------------
    # Model Validation
    # -------------------------------------------------------------------------
    "time_series_cv": {
        "title": "Time-Series Cross-Validation",
        "content": """
            Unlike standard cross-validation, <strong>time-series CV</strong> respects 
            temporal order: we always train on past data and test on future data.
            <br><br>
            <strong>Our approach:</strong>
            <ul>
                <li><strong>Training window:</strong> 52 weeks (1 year of history)</li>
                <li><strong>Test window:</strong> 13 weeks (1 quarter ahead)</li>
                <li><strong>Roll forward:</strong> 13 weeks between folds</li>
            </ul>
            This tests whether the model can predict <em>future</em> revenue—not just 
            fit historical patterns. A model that passes this test will generalize 
            to real planning decisions.
            <br><br>
            <strong>Key metric:</strong> CV MAPE &lt; 15% means predictions are typically 
            within 15% of actual revenue.
        """,
        "persona": ["data_scientist"],
        "depth": "technical"
    },
    
    "bootstrap_ci": {
        "title": "Bootstrap Confidence Intervals",
        "content": """
            <strong>Bootstrap</strong> is a resampling technique to quantify uncertainty 
            without assuming a specific statistical distribution.
            <br><br>
            <strong>How it works:</strong>
            <ol>
                <li>Randomly resample the data with replacement (100 times)</li>
                <li>Re-fit the model on each "fake" dataset</li>
                <li>Collect the distribution of ROI estimates</li>
                <li>5th and 95th percentiles → 90% confidence interval</li>
            </ol>
            <strong>Why bootstrap for MMM:</strong>
            <ul>
                <li>No normality assumption (errors are often skewed)</li>
                <li>Handles autocorrelation in time series</li>
                <li>Works with nonlinear transforms (adstock, saturation)</li>
            </ul>
            Note: We keep hyperparameters fixed during bootstrap to isolate 
            coefficient uncertainty.
        """,
        "persona": ["data_scientist"],
        "depth": "technical"
    },
    
    # -------------------------------------------------------------------------
    # Recommendations & Optimization
    # -------------------------------------------------------------------------
    "budget_optimizer": {
        "title": "Why These Recommendations?",
        "content": """
            The budget optimizer finds the allocation that <strong>maximizes predicted 
            revenue</strong> while respecting realistic constraints.
            <br><br>
            <strong>How it works:</strong>
            <ol>
                <li>Starts with current allocation as baseline</li>
                <li>Uses marginal ROI to identify reallocation opportunities</li>
                <li>Shifts budget from saturated to efficient channels</li>
                <li>Applies ±30% constraint (no channel changes by more than 30%)</li>
            </ol>
            <strong>Why constraints matter:</strong>
            <ul>
                <li>Prevents unrealistic "put 100% in LinkedIn" recommendations</li>
                <li>Respects organizational change capacity</li>
                <li>Maintains channel diversification for risk management</li>
            </ul>
        """,
        "persona": ["cmo", "regional_lead"],
        "depth": "tactical"
    },
    
    "recommendation_confidence": {
        "title": "Recommendation Confidence Levels",
        "content": """
            Recommendations are tagged with confidence levels based on the 
            <strong>statistical reliability</strong> of underlying estimates.
            <br><br>
            <strong>High Confidence</strong> (green):
            <ul>
                <li>Both source and target channels have narrow CIs</li>
                <li>Marginal ROI difference is statistically significant</li>
                <li>Predicted lift is likely to materialize</li>
            </ul>
            <strong>Medium Confidence</strong> (amber):
            <ul>
                <li>One channel has wider uncertainty</li>
                <li>Recommendation is directionally correct but magnitude uncertain</li>
            </ul>
            <strong>Low Confidence</strong> (red):
            <ul>
                <li>Wide CIs on both channels</li>
                <li>Recommendation may not yield expected results</li>
                <li>Consider gathering more data before acting</li>
            </ul>
        """,
        "persona": ["cmo", "regional_lead"],
        "depth": "tactical"
    },
    
    # -------------------------------------------------------------------------
    # Data & Methodology
    # -------------------------------------------------------------------------
    "data_sources": {
        "title": "Where Does This Data Come From?",
        "content": """
            The MMM integrates data from three enterprise systems:
            <br><br>
            <strong>Sprinklr</strong> (Ad Platform):
            <ul>
                <li>Daily spend, impressions, clicks by campaign</li>
                <li>23 marketing channels across Social, Search, Display, Video</li>
            </ul>
            <strong>Salesforce</strong> (CRM):
            <ul>
                <li>Pipeline opportunities linked to campaigns</li>
                <li>Win rates and deal sizes for calibration</li>
            </ul>
            <strong>SAP</strong> (ERP):
            <ul>
                <li>Booked revenue (the ground truth for ROI)</li>
                <li>Includes distributor sales that can't be directly attributed</li>
            </ul>
            Data is aggregated to weekly grain at Channel × Region × Product level.
        """,
        "persona": ["cmo", "regional_lead", "data_scientist"],
        "depth": "overview"
    },
    
    "model_controls": {
        "title": "What Are Control Variables?",
        "content": """
            <strong>Control variables</strong> account for factors that affect revenue 
            but aren't marketing—ensuring we isolate true marketing impact.
            <br><br>
            <strong>Controls in this model:</strong>
            <ul>
                <li><strong>Seasonality</strong> (Fourier terms): Captures Q4 holiday spikes, 
                    summer slowdowns, etc.</li>
                <li><strong>PMI Index</strong>: Economic indicator of industrial activity. 
                    When PMI drops, demand falls regardless of marketing.</li>
                <li><strong>Competitor SOV</strong>: Share of voice. When competitors 
                    increase advertising, our effectiveness may decrease.</li>
            </ul>
            Without controls, the model would incorrectly attribute seasonal revenue 
            to whatever channel happened to be active during that period.
        """,
        "persona": ["data_scientist"],
        "depth": "technical"
    },
    
    "ridge_regression": {
        "title": "Why Ridge Regression?",
        "content": """
            The model uses <strong>Ridge Regression</strong> (L2 regularization) with 
            positive coefficient constraints.
            <br><br>
            <strong>Why Ridge?</strong>
            <ul>
                <li><strong>Prevents overfitting</strong>: L2 penalty shrinks coefficients, 
                    avoiding wild extrapolations</li>
                <li><strong>Handles multicollinearity</strong>: Marketing channels often 
                    move together; Ridge stabilizes estimates</li>
                <li><strong>Positive constraints</strong>: Ensures "more spend ≠ less revenue"—
                    a channel can only help or be neutral</li>
            </ul>
            <strong>Ridge alpha = 1.0</strong> provides moderate regularization, 
            balancing fit vs. stability.
        """,
        "persona": ["data_scientist"],
        "depth": "technical"
    },
    
    "nevergrad_optimization": {
        "title": "How Nevergrad Finds Optimal Parameters",
        "content": """
            <strong>Nevergrad</strong> is Meta's derivative-free optimization library. 
            We use it because MMM hyperparameters don't have clean gradients.
            <br><br>
            <strong>The problem:</strong> Find the best (θ, α, γ) for each channel 
            that minimizes prediction error—but these parameters are nested inside 
            nonlinear transforms with no closed-form solution.
            <br><br>
            <strong>The solution:</strong> TwoPointsDE algorithm (evolutionary):
            <ol>
                <li>Start with population of candidate parameter sets</li>
                <li>Evaluate each by fitting Ridge and measuring error</li>
                <li>Breed best performers, mutate, repeat</li>
                <li>After 500 iterations, return best parameters found</li>
            </ol>
            This finds near-optimal parameters without gradient computation.
        """,
        "persona": ["data_scientist"],
        "depth": "technical"
    }
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_explanation(key: str) -> dict:
    """Get explanation content by key. Returns empty dict if not found."""
    return EXPLANATIONS.get(key, {})


def get_explanations_for_persona(persona: str) -> list:
    """Get all explanations relevant to a specific persona."""
    return [
        (key, exp) for key, exp in EXPLANATIONS.items() 
        if persona in exp.get("persona", [])
    ]


def get_explanations_by_depth(depth: str) -> list:
    """Get all explanations at a specific depth level."""
    return [
        (key, exp) for key, exp in EXPLANATIONS.items() 
        if exp.get("depth") == depth
    ]


def render_explanation_expander(key: str, st_module):
    """
    Render an explanation as a Streamlit expander.
    
    Usage:
        from utils.explanations import render_explanation_expander
        render_explanation_expander("confidence_intervals", st)
    """
    exp = get_explanation(key)
    if not exp:
        return
    
    with st_module.expander(f"{exp['title']}", expanded=False):
        st_module.markdown(exp['content'], unsafe_allow_html=True)


# =============================================================================
# TOOLTIP CONTENT (shorter versions for hover)
# =============================================================================

TOOLTIPS = {
    "roi": "Return on Investment: Revenue generated per dollar spent",
    "roas": "Return on Ad Spend: Same as ROI in marketing context",
    "marginal_roi": "Revenue from the NEXT dollar spent (not historical average)",
    "ci_90": "90% Confidence Interval: Range where true value likely falls",
    "significant": "Statistically significant: CI excludes breakeven (1.0x)",
    "adstock_decay": "How quickly advertising effect fades (0=instant, 1=permanent)",
    "saturation_gamma": "Spend level where response is 50% of maximum",
    "saturation_alpha": "Shape of S-curve (higher = steeper transition)",
    "cv_mape": "Cross-validation error: Avg % difference from actual revenue",
    "r_squared": "Model fit: % of revenue variance explained (higher = better)",
    "efficient_zone": "Marginal ROI > 1.5x: Strong returns on additional spend",
    "diminishing_zone": "Marginal ROI 0.8-1.5x: Returns declining but still positive",
    "saturated_zone": "Marginal ROI < 0.8x: Diminishing returns, consider reallocation"
}


def get_tooltip(key: str) -> str:
    """Get tooltip text by key."""
    return TOOLTIPS.get(key, "")

