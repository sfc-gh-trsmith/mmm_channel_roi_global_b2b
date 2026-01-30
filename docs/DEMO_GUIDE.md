# Global B2B Marketing Mix Modeling (MMM) & ROI Engine
## Complete Demo Guide for NotebookLM

---

# PART 1: THE BUSINESS PROBLEM

## Why This Exists

For global B2B enterprises—think multinational manufacturers selling industrial equipment, healthcare supplies, and consumer products—marketing data is fragmented across three incompatible systems:

1. **Sprinklr (Ad Platform)**: Tracks impressions, clicks, and spend for LinkedIn, Google Ads, and programmatic display campaigns
2. **Salesforce (CRM)**: Contains pipeline opportunities with deal stages, amounts, and close dates
3. **SAP (ERP)**: Records actual invoiced revenue—the "ground truth" of what was sold

**The fundamental problem**: These systems don't talk to each other. A marketing leader who spends $500,000 on LinkedIn campaigns in January cannot trace that investment to the $3.2M in distributor revenue that shows up in SAP nine months later.

## The 9-Month Attribution Gap

B2B sales cycles are brutally long. Consider this timeline:

- **Week 1**: A procurement manager at a manufacturing plant sees a LinkedIn video ad for industrial safety equipment
- **Week 3**: Same manager downloads a whitepaper after seeing a retargeting ad
- **Week 8**: Manager attends a webinar, enters Salesforce as a Marketing Qualified Lead
- **Week 15**: Sales rep schedules discovery call, creates Opportunity in Salesforce
- **Week 28**: Proposal submitted, enters "Negotiation" stage
- **Week 36**: Contract signed, marked "Closed Won" in Salesforce
- **Week 42**: First PO processed, inventory shipped
- **Week 48**: Invoice posted in SAP as recognized revenue

The original LinkedIn impression and the SAP invoice are separated by **nearly a year** and live in completely different systems. Traditional "last-click attribution" would credit none of this revenue to LinkedIn—it would either credit the sales rep's direct outreach or show no marketing attribution at all.

## The Cost of Blind Budgeting

Without proper attribution, marketing leaders resort to "peanut-butter spreading"—allocating equal percentages of budget to each channel regardless of performance. This causes:

- **$2.4M in annual waste per business unit** from over-investing in saturated channels
- **Missed opportunities** to scale channels with proven ROI headroom
- **Political budget battles** where the loudest voice wins, not the best data
- **No ability to defend marketing ROI** when the CFO asks for justification

---

# PART 2: THE SOLUTION ARCHITECTURE

## What Marketing Mix Modeling Actually Does

Marketing Mix Modeling (MMM) uses **statistical regression** to estimate the causal effect of marketing spend on revenue. Unlike pixel-based attribution (which breaks with privacy regulations and long sales cycles), MMM works with aggregate data:

**The Core Question**: *"For every dollar we spend on LinkedIn vs. Google vs. Display, how much revenue do we get back?"*

**What Makes MMM Different from Last-Click**:

| Aspect | Last-Click Attribution | Marketing Mix Modeling |
|--------|----------------------|------------------------|
| Data Source | Individual user tracking pixels | Aggregate spend and revenue by week |
| Time Handling | Credits final touchpoint before conversion | Models time-delayed effects with "adstock" |
| Diminishing Returns | Ignores saturation | Explicitly models S-curves |
| Privacy | Requires cookies/tracking | Works with aggregated, privacy-safe data |
| Sales Cycle | Works for immediate conversions | Handles 6-18 month B2B cycles |

## The Four-Part Technical Framework

This solution implements four core techniques from marketing science and econometrics:

### 1. Geometric Adstock (The Carryover Effect)

**The Problem**: A $100,000 LinkedIn campaign in Week 1 doesn't just affect Week 1. In B2B, someone sees an ad, thinks about it, researches competitors, attends a webinar, and converts weeks or months later. The advertising effect "carries over" into future periods.

**The Solution**: Spread each week's spend across future weeks with exponential decay. The formula is recursive:

```
Adstocked_Spend[t] = Raw_Spend[t] + θ × Adstocked_Spend[t-1]
```

Where θ (theta) is the decay rate, typically between 0 and 1.

**Concrete Example** (θ = 0.7):

| Week | Raw Spend | Adstocked (Effective) Spend |
|------|-----------|---------------------------|
| 1 | $100,000 | $100,000 |
| 2 | $0 | $70,000 (70% carryover) |
| 3 | $0 | $49,000 (70% × $70k) |
| 4 | $0 | $34,300 (70% × $49k) |
| 5 | $0 | $24,010 |
| 6 | $0 | $16,807 |

After 6 weeks with zero new spend, the original $100k still contributes nearly $17k of "effective" spend due to lingering brand awareness.

**Decay Rate by Channel** (learned automatically by the model):
- **LinkedIn B2B**: θ = 0.7–0.9 (long consideration cycle, effects persist 6-8 weeks)
- **Paid Search**: θ = 0.1–0.3 (immediate intent, fast decay—people search when ready to buy)
- **Display/Programmatic**: θ = 0.4–0.6 (awareness building, medium decay)
- **TV/Brand Campaigns**: θ = 0.85–0.95 (very long brand effects)

**The Half-Life Interpretation**: The decay rate determines how long it takes for the effect to halve. With θ = 0.7, the half-life is about 2 weeks. With θ = 0.9, the half-life is about 6.5 weeks.

### 2. Hill Saturation (The Diminishing Returns Curve)

**The Problem**: Doubling spend doesn't double revenue. At some point, you've reached everyone in your target audience who will ever respond. The 10th impression to the same person has near-zero incremental value.

**The Solution**: Apply an S-curve transformation (the Hill function from pharmacology) that captures this saturation:

```
Response = Spend^α / (Spend^α + γ^α)
```

Where:
- **α (alpha)**: Controls the steepness of the S-curve. Higher α means sharper transition from "efficient" to "saturated"
- **γ (gamma)**: The "half-saturation point"—the spend level where response is exactly 50% of maximum

**Visual Representation**:

```
Response │              ●●●●●●●●●●●●  ← Saturation Zone (mROI < $0.80)
   ▲     │         ●●●●●               Each additional dollar returns less
   │     │      ●●●                    than a dollar—money is being wasted
   │     │    ●●
   │     │  ●●                       ← Efficient Zone (mROI > $1.50)
   │     │●●                           Every dollar returns $1.50+
   │     │                             Strong investment territory
   └─────┴──────────────────────────► Spend
         0      γ (half-saturation)
```

**Why This Matters for Budget Decisions**: A channel can have excellent average ROI (total return ÷ total spend = 3x) but terrible marginal ROI (the *next* dollar only returns $0.50). Budget decisions should be based on marginal ROI—where should the *next* dollar go, not where did the last dollar go.

### 3. Nevergrad Evolutionary Optimization (Finding the Best Parameters)

**The Problem**: With 20 marketing channels and 3 parameters each (θ, α, γ), we have 60 parameters to optimize. Grid search is impossible—60^10 combinations would take longer than the age of the universe. Traditional gradient descent doesn't work because the objective function isn't smooth (it involves re-training a regression model at each step).

**The Solution**: Use an evolutionary algorithm called TwoPointsDE (Two-Points Differential Evolution) from Meta's Nevergrad library:

1. **Initialize**: Start with a "population" of 50 random parameter guesses
2. **Evaluate**: Test each guess by training the regression and computing R²
3. **Breed**: Create new guesses by combining successful parents: `new = parent_a + F × (parent_b - parent_c)`
4. **Select**: Keep the best performers, discard the worst
5. **Repeat**: 500 iterations until convergence

**Why Evolutionary Works Here**: Unlike gradient-based methods, evolutionary algorithms don't need smooth objectives. They work by trial-and-error guided by selection pressure—exactly what's needed when the objective involves complex nested computations.

**The Budget Parameter**: "Budget" in Nevergrad means "number of function evaluations," not dollars. We use 500 iterations, which typically converges in 3-5 minutes.

### 4. Ridge Regression with Positive Constraints

**The Problem**: Marketing channels are correlated—LinkedIn and Display both spike during Q4 holiday campaigns. Standard OLS (Ordinary Least Squares) regression produces wild, unstable coefficients when predictors are correlated. You might get "LinkedIn = +$5M" and "Display = -$3M" when the truth is both contribute positively.

**The Solution**: Ridge Regression adds an L2 penalty that shrinks coefficients toward zero:

```
Minimize: ||y - Xβ||² + λ||β||²
```

Where λ (lambda) controls the strength of regularization. Higher λ = more shrinkage = more stable but potentially biased coefficients.

**Why Ridge Instead of Lasso**: Lasso (L1 penalty) would zero out some channels entirely. That's great for feature selection but bad for marketing—we want *every* channel's contribution estimated, even small ones. Ridge shrinks but never zeros.

**The Positive Constraint Trick**: Economically, marketing should never *hurt* revenue. A negative coefficient means the model thinks "more LinkedIn spend = less revenue," which is usually a sign of multicollinearity or data issues, not reality. We add a penalty term that pushes negative media coefficients toward zero.

---

# PART 3: THE DATA ARCHITECTURE

## Schema Organization

The solution uses a layered schema architecture following modern data engineering best practices:

| Schema | Purpose | Example Tables |
|--------|---------|----------------|
| **RAW** | Landing zone for source data | `SPRINKLR_DAILY`, `SFDC_OPPORTUNITIES`, `SAP_ACTUALS` |
| **ATOMIC** | Normalized, clean data with SCD2 versioning | `GEOGRAPHY`, `PRODUCT_CATEGORY`, `MARKETING_MEDIA_SPEND` |
| **DIMENSIONAL** | Flattened views for analytics | `V_MMM_INPUT_WEEKLY`, `DIM_GEOGRAPHY_HIERARCHY` |
| **MMM** | Model outputs and analytics mart | `MODEL_RESULTS`, `RESPONSE_CURVES` |

## The Dimensional Hierarchies

### Geography Hierarchy (75+ Countries)

Three-level self-referential hierarchy:

```
Super-Region → Region → Country

AMERICAS
├── North America
│   ├── USA
│   ├── Canada
│   └── Mexico
├── Latin America North
│   ├── Guatemala
│   ├── Costa Rica
│   └── Colombia
└── Latin America South
    ├── Brazil
    ├── Argentina
    └── Chile

EMEA
├── Western Europe
│   ├── Germany
│   ├── France
│   └── UK
├── Central & Eastern Europe
│   ├── Poland
│   └── Czech Republic
└── Middle East
    ├── UAE
    └── Saudi Arabia

ASIA PACIFIC
├── Greater China
│   ├── China
│   └── Hong Kong
├── Japan & Korea
├── ASEAN (Singapore, Thailand, Vietnam...)
└── ANZ (Australia, New Zealand)
```

### Product Hierarchy (23 Product Lines)

Three-level hierarchy across 4 business segments:

```
Segment → Division → Category

SAFETY & INDUSTRIAL
├── Abrasives & Surface
│   ├── Abrasives
│   ├── Compounds & Polishes
│   └── Coatings
├── Adhesives & Tapes
│   ├── Adhesives/Sealants/Fillers
│   ├── Tapes
│   └── Films & Sheeting
├── Safety Solutions
│   ├── Personal Protective Equipment
│   └── Filtration & Separation
└── Industrial Products
    ├── Tools & Equipment
    └── Lubricants

TRANSPORTATION & ELECTRONICS
├── Automotive
│   ├── Automotive Parts & Hardware
│   └── Advanced Materials
└── Electronics
    └── Electronics Materials & Components

HEALTHCARE
└── Medical Solutions
    ├── Medical
    └── Lab Supplies & Testing

CONSUMER
└── Consumer Home
    ├── Home
    └── Office Supplies
```

### Marketing Channels (23 Channels)

```
Channel Type → Channel

SOCIAL
├── LinkedIn
├── LinkedIn Video
├── LinkedIn InMail
├── Facebook
└── Instagram

SEARCH
├── Google Search
└── Bing Search

PROGRAMMATIC
├── Google Display
├── Programmatic Display
├── Programmatic Video
└── Connected TV

VIDEO
└── YouTube

DIRECT
├── Email
├── Direct Mail
└── Trade Publications

EVENTS
├── Webinars
└── Trade Shows
```

## The Key Views for Model Training

### V_MMM_INPUT_WEEKLY

This is the primary training view that aggregates daily data to weekly grain with all dimensional attributes:

```sql
SELECT
    WEEK_START,
    SUPER_REGION_NAME,
    REGION_NAME,
    COUNTRY_NAME,
    SEGMENT_NAME,
    DIVISION_NAME,
    CATEGORY_NAME,
    CHANNEL_CODE,
    -- Media metrics (X variables)
    SUM(SPEND) AS SPEND,
    SUM(IMPRESSIONS) AS IMPRESSIONS,
    SUM(CLICKS) AS CLICKS,
    -- Revenue (Y variable)
    SUM(REVENUE) AS REVENUE,
    -- Control variables
    AVG(PMI_INDEX) AS PMI,
    AVG(COMPETITOR_SOV) AS COMPETITOR_SOV
FROM aggregated_data
GROUP BY dimensions
```

**Why Weekly Grain**: Daily data is too noisy for MMM—campaigns don't cause same-day revenue. Weekly smooths out day-of-week effects while preserving seasonality patterns.

### Revenue Attribution Logic

The critical insight: Revenue in SAP doesn't have a direct "campaign ID" field. The attribution chain is:

```
Revenue (SAP Invoice)
    → links to Opportunity (Salesforce)
        → links to Campaign (Sprinklr)
            → has Channel assignment
```

So revenue is attributed to channels through the CRM opportunity that was sourced by the campaign. This captures the full B2B funnel.

---

# PART 4: THE TRAINING PIPELINE

## Cell-by-Cell Walkthrough of the MMM Training Notebook

### Cell 0: Package Installation

```python
import os
import sys
packages = ["nevergrad"]
for pkg in packages:
    os.system(f"{sys.executable} -m pip install {pkg} -q")
```

Nevergrad isn't pre-installed in Snowflake Notebooks, so we install it at runtime. This requires an External Access Integration for PyPI connectivity.

### Cell 1: Configuration

```python
@dataclass
class MMMConfig:
    # Data sources
    input_view: str = "DIMENSIONAL.V_MMM_INPUT_WEEKLY"
    output_table: str = "ATOMIC.MMM_MODEL_RESULT"
    
    # Model granularity
    geo_level: str = "GLOBAL"      # GLOBAL, SUPER_REGION, REGION, COUNTRY
    product_level: str = "SEGMENT" # SEGMENT, DIVISION, CATEGORY
    
    # Optimization
    nevergrad_budget: int = 500    # Evolutionary iterations
    ridge_alpha: float = 10.0      # L2 penalty (higher = more regularization)
    
    # Cross-validation
    cv_train_weeks: int = 52       # 1 year training window
    cv_test_weeks: int = 13        # 1 quarter holdout
    
    # Bootstrap
    n_bootstrap: int = 100         # Resample iterations
    confidence_level: float = 0.90 # 90% confidence intervals
    
    # Budget optimizer
    budget_change_limit: float = 0.30  # ±30% per channel
```

**Key Design Decisions**:

- **GLOBAL geo_level**: Aggregates all regions to maximize sample size per channel. Use SUPER_REGION only if you have 50+ weeks of data per channel-region combination.
- **ridge_alpha = 10.0**: Higher than default (1.0) because B2B data is sparse and noisy.
- **52-week training, 13-week test**: Captures full seasonality in training, evaluates on one quarter.

### Cell 2: Data Loading

```python
session = get_active_session()
df_raw = session.table(config.input_view).to_pandas()

# Column mapping from view to model expectations
column_mapping = {
    'SUPER_REGION_NAME': 'SUPER_REGION',
    'CHANNEL_CODE': 'CHANNEL',
    'AVG_PMI': 'PMI_INDEX',
    ...
}
df_raw = df_raw.rename(columns=column_mapping)
```

The view uses descriptive column names (`SUPER_REGION_NAME`), but the model expects simpler names (`SUPER_REGION`).

### Cell 3: Feature Engineering

**Composite Keys**: We model at the Channel × Region × Product level. Each combination gets its own coefficient:

```python
df['CHANNEL_KEY'] = (
    df['CHANNEL'].astype(str) + '_' + 
    df[geo_col].astype(str) + '_' + 
    df[prod_col].astype(str)
)
# Example: "LINKEDIN_EMEA_SI" = LinkedIn in EMEA for Safety & Industrial
```

**Fourier Seasonality**: Instead of 52 dummy variables (one per week), we use smooth sine/cosine waves:

```python
for k in [1, 2, 3]:
    df[f'SIN_{k}'] = np.sin(2 * np.pi * k * df['WEEK_OF_YEAR'] / 52)
    df[f'COS_{k}'] = np.cos(2 * np.pi * k * df['WEEK_OF_YEAR'] / 52)
```

This captures:
- **k=1**: Annual cycle (one peak per year)
- **k=2**: Semi-annual (captures "Q2 and Q4 different from Q1 and Q3")
- **k=3**: Quarterly patterns

**Why Fourier Instead of Dummies**: Six features capture smooth seasonality. 52 dummies would overfit and can't extrapolate to unseen weeks.

**Linear Trend**: Captures organic growth unrelated to marketing:

```python
df['TREND'] = (df['WEEK_START'] - df['WEEK_START'].min()).dt.days / 365.25
```

Without this, the model would attribute organic growth to whichever channel happened to scale up.

### Cell 4: Adstock and Saturation Functions

```python
def geometric_adstock(x: np.ndarray, theta: float) -> np.ndarray:
    """Recursive adstock: x_eff[t] = x[t] + theta * x_eff[t-1]"""
    x_adstocked = np.zeros_like(x)
    x_adstocked[0] = x[0]
    for t in range(1, len(x)):
        x_adstocked[t] = x[t] + theta * x_adstocked[t-1]
    return x_adstocked

def hill_saturation(x: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    """Hill function: x^α / (x^α + γ^α)"""
    x = np.maximum(x, 0)
    gamma = max(gamma, 1e-10)  # Avoid division by zero
    return (x ** alpha) / (x ** alpha + gamma ** alpha)
```

The order matters: **Adstock first, then Saturation**. Adstock spreads the spend signal over time; saturation then applies diminishing returns to the accumulated signal.

### Cell 5: Pivoting to Wide Format

The model needs one row per week, one column per channel:

```
Before (long):
WEEK  | CHANNEL_KEY        | SPEND   | REVENUE
W1    | LINKEDIN_EMEA_SI   | 50000   | 100000
W1    | GOOGLE_EMEA_SI     | 30000   | 100000
W2    | LINKEDIN_EMEA_SI   | 45000   | 105000

After (wide):
WEEK | LINKEDIN_EMEA_SI | GOOGLE_EMEA_SI | ... | TOTAL_REVENUE
W1   | 50000            | 30000          | ... | 100000
W2   | 45000            | 25000          | ... | 105000
```

Channels with less than $1,000 total spend are dropped—not enough signal to estimate reliably.

### Cell 6: Time-Series Cross-Validation

**The Critical Distinction**: Standard k-fold CV randomly shuffles data. This lets the model "peek" at Week 100's data when predicting Week 50—giving falsely optimistic metrics.

Time-series CV respects temporal order:

```
Fold 1: Train [Week 1-52]   → Test [Week 53-65]   (predict Q1 2024)
Fold 2: Train [Week 14-65]  → Test [Week 66-78]   (predict Q2 2024)
Fold 3: Train [Week 27-78]  → Test [Week 79-91]   (predict Q3 2024)
```

This mimics real use: "Given everything up to today, how well can we predict next quarter?"

**Quality Interpretation**:

| CV MAPE | Quality | Implication |
|---------|---------|-------------|
| < 10% | Excellent | Model is highly predictive |
| 10-20% | Good | Suitable for budget optimization |
| 20-30% | Acceptable | Directional insights only |
| > 30% | Poor | Investigate data quality or model specification |

### Cell 7: Nevergrad Optimization

```python
class MMMOptimizer:
    def _objective(self, flat_params):
        params = self._decode_params(flat_params)
        X_media_trans = apply_media_transformations(self.X_media, params, self.channels)
        
        # Train Ridge regression
        model = Ridge(alpha=self.config.ridge_alpha)
        model.fit(X_scaled, self.y)
        
        # Calculate R² (maximize) = minimize (1 - R²)
        r2 = r2_score(self.y, model.predict(X_scaled))
        
        # Penalize negative coefficients (economically invalid)
        media_coefs = model.coef_[:len(self.channels)]
        negative_penalty = np.sum(np.minimum(media_coefs, 0) ** 2) * 10
        
        return (1 - r2) + negative_penalty
```

**The Sigmoid Reparametrization**: Instead of searching [0, 0.95] for theta directly, we search [-5, 5] and apply sigmoid to map to valid ranges. This avoids boundary issues and makes the search space smoother.

### Cell 8: Final Model Training

We train two versions:

1. **In-Sample**: Train on ALL data, predict on ALL data. R² will be high but optimistic.
2. **Cross-Validation**: Train on past, predict on future. Shows realistic accuracy.

**Healthy Gap**: In-sample R² ≈ CV R² + 0.05-0.10. If the gap is larger, the model is overfitting.

### Cell 9: Bootstrap Confidence Intervals

Standard error formulas assume normality and independence—conditions violated by MMM. Bootstrap provides distribution-free uncertainty:

```python
for b in range(100):
    # Resample with replacement
    boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
    X_boot, y_boot = X_full.iloc[boot_idx], y.iloc[boot_idx]
    
    # Re-fit model
    model.fit(scaler.fit_transform(X_boot), y_boot)
    
    # Calculate ROI for each channel
    for ch in channels:
        roi = contribution / spend
        roi_samples[ch].append(roi)

# 90% CI = 5th to 95th percentile
ci_lower = np.percentile(roi_samples[ch], 5)
ci_upper = np.percentile(roi_samples[ch], 95)
```

**Output Interpretation**: "LinkedIn ROI = 3.2x [2.8, 3.6] at 90% confidence" means we're 90% confident the true ROI is between 2.8x and 3.6x.

**IS_SIGNIFICANT Flag**: True if the entire CI is above zero (or above 1.0 for breakeven).

### Cell 10: Response Curves and Marginal ROI

**Average ROI vs. Marginal ROI**:

- **Average ROI**: Total contribution / Total spend = "What did we get back per dollar historically?"
- **Marginal ROI**: Derivative of response curve at current spend = "What will we get for the *next* dollar?"

A channel with 5x average ROI might be saturated with 0.5x marginal ROI. Budget decisions should use marginal, not average.

**Efficiency Zones**:

| Marginal ROI | Zone | Implication |
|--------------|------|-------------|
| > 1.5 | EFFICIENT | Strong investment territory |
| 0.8 - 1.5 | DIMINISHING | Still positive but flattening |
| < 0.8 | SATURATED | Reallocate to other channels |

### Cell 11: Budget Optimizer

Given learned response curves, find the optimal reallocation:

```
MAXIMIZE: Total predicted revenue = Σ f_i(spend_i)
SUBJECT TO:
  Σ spend_i = current_total_budget    (budget neutral)
  (1-0.3) × current_i ≤ spend_i ≤ (1+0.3) × current_i  (±30%)
  spend_i ≥ 0
```

**Why Constraints Matter**: Without them, the optimizer says "put 100% in the highest-marginal-ROI channel." But:
- CMOs can't pivot 100% of budget in one quarter
- Vendor contracts have minimum commitments
- Channel inventory is finite (LinkedIn can only show so many ads)

**The Economic Principle**: The optimal allocation **equalizes marginal ROI across all channels** (subject to constraints). If LinkedIn has mROI = 3.0 and Display has mROI = 1.5, shift budget until they converge.

### Cells 12-13: Save to Snowflake

Outputs are saved to three tables:

1. **MMM.MODEL_RESULTS**: Channel-level ROI with confidence intervals, learned parameters
2. **MMM.RESPONSE_CURVES**: 100-point curves per channel with CI bands
3. **MMM.MODEL_METADATA**: Model configuration and quality metrics

The model is also registered to the **Snowflake Model Registry** for version tracking and comparison.

---

# PART 5: THE SUPPLEMENTARY NOTEBOOK

## 02_snowflake_ml_features.ipynb: SQL Features and FORECAST Baseline

This notebook demonstrates additional Snowflake ML capabilities:

### SQL Feature Engineering

Instead of computing Fourier seasonality in Python, we can push it to SQL:

```sql
SELECT 
    WEEK_START,
    DATEDIFF('day', '2022-01-01', WEEK_START) / 365.25 AS TREND,
    SIN(2 * PI() * 1 * WEEKOFYEAR(WEEK_START) / 52) AS SIN_1,
    COS(2 * PI() * 1 * WEEKOFYEAR(WEEK_START) / 52) AS COS_1,
    SIN(2 * PI() * 2 * WEEKOFYEAR(WEEK_START) / 52) AS SIN_2,
    COS(2 * PI() * 2 * WEEKOFYEAR(WEEK_START) / 52) AS COS_2,
    CASE WHEN MONTH(WEEK_START) BETWEEN 1 AND 3 THEN 1 ELSE 0 END AS Q1_FLAG
FROM weekly_data
```

**Why SQL Over Python**: Scales to any data size (Snowflake handles compute), ensures consistency between training and serving, reduces data transfer.

### FORECAST Baseline

Snowflake's built-in FORECAST function provides a naive time-series baseline:

```sql
CREATE SNOWFLAKE.ML.FORECAST MMM.REVENUE_FORECAST_MODEL (
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'MMM.V_WEEKLY_REVENUE_FOR_FORECAST'),
    TIMESTAMP_COLNAME => 'WEEK_START',
    TARGET_COLNAME => 'TOTAL_REVENUE'
);
```

**Why MMM Outperforms FORECAST**: FORECAST just extrapolates historical patterns. It doesn't know:
- That a $500K LinkedIn campaign is launching next quarter
- That competitor share-of-voice is increasing
- The causal relationship between spend and revenue

MMM captures these relationships explicitly.

### When to Use Feature Store

For this demo, SQL views are sufficient. Feature Store adds value when:
- Multiple models share the same features (propensity, churn, LTV all using customer_tenure)
- You need point-in-time correctness for training vs. serving
- Feature governance and lineage are critical
- Data science teams need to share curated features

---

# PART 6: THE SETUP PROCESS

## Step-by-Step Deployment

### 1. Account Setup (01_account_setup.sql)

Creates Snowflake account objects:

```sql
-- Create dedicated role
CREATE ROLE IF NOT EXISTS PROJECT_ROLE;

-- Create database
CREATE DATABASE IF NOT EXISTS GLOBAL_B2B_MMM;

-- Create warehouse (auto-suspend after 60s)
CREATE WAREHOUSE IF NOT EXISTS PROJECT_WH
    WAREHOUSE_SIZE = 'XSMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE;

-- Create compute pool for notebooks
CREATE COMPUTE POOL IF NOT EXISTS PROJECT_COMPUTE_POOL
    MIN_NODES = 1
    MAX_NODES = 1
    INSTANCE_FAMILY = CPU_X64_S;

-- External access for PyPI (nevergrad installation)
CREATE EXTERNAL ACCESS INTEGRATION PYPI_ACCESS_INTEGRATION
    ALLOWED_NETWORK_RULES = (PYPI_NETWORK_RULE)
    ENABLED = TRUE;
```

### 2. Schema Setup (02_schema_setup.sql)

Creates all tables and initial views:

**Dimension Tables**:
- `GEOGRAPHY` (self-referential hierarchy)
- `PRODUCT_CATEGORY` (self-referential hierarchy)
- `ORGANIZATION` (self-referential hierarchy)
- `MARKETING_CHANNEL` (flat)
- `CURRENCY` (flat)

**Fact Tables**:
- `MARKETING_CAMPAIGN` (campaign metadata with FKs to all dimensions)
- `MARKETING_MEDIA_SPEND` (daily spend metrics)
- `MARKET_INDICATOR` (weekly macro controls)
- `OPPORTUNITY` (CRM pipeline)
- `ACTUAL_FINANCIAL_RESULT` (ERP revenue)
- `MMM_MODEL_RESULT` (model outputs)

### 3. Load Data (03_load_data.sql)

Loads data from stage to tables:

```sql
-- Copy from stage to RAW tables
COPY INTO SPRINKLR_DAILY
FROM @ATOMIC.DATA_STAGE/sprinklr_spend.csv
FILE_FORMAT = (FORMAT_NAME = ATOMIC.CSV_FORMAT);

-- Transform and load to ATOMIC
INSERT OVERWRITE INTO ATOMIC.MEDIA_SPEND_DAILY
SELECT DATE, CAMPAIGN_ID, CHANNEL, SPEND_AMT, IMPRESSIONS, CLICKS, VIDEO_VIEWS_50
FROM SPRINKLR_DAILY;
```

### 4. Dimensional Views (03_dimensional_views.sql)

Creates flattened hierarchy views:

```sql
-- Flatten geography: Super-Region → Region → Country
CREATE VIEW DIM_GEOGRAPHY_HIERARCHY AS
SELECT 
    g.GEOGRAPHY_NAME AS COUNTRY_NAME,
    r.GEOGRAPHY_NAME AS REGION_NAME,
    sr.GEOGRAPHY_NAME AS SUPER_REGION_NAME
FROM GEOGRAPHY g
LEFT JOIN GEOGRAPHY r ON g.PARENT_GEOGRAPHY_ID = r.GEOGRAPHY_ID
LEFT JOIN GEOGRAPHY sr ON r.PARENT_GEOGRAPHY_ID = sr.GEOGRAPHY_ID
WHERE g.GEOGRAPHY_TYPE = 'COUNTRY';
```

And the master MMM input view:

```sql
CREATE VIEW V_MMM_INPUT_WEEKLY AS
WITH WEEKLY_SPEND AS (...),
     WEEKLY_REVENUE AS (...),
     WEEKLY_INDICATORS AS (...)
SELECT
    WEEK_START,
    SUPER_REGION_NAME,
    CHANNEL_CODE,
    SUM(SPEND) AS SPEND,
    SUM(REVENUE) AS REVENUE,
    AVG(PMI) AS PMI_INDEX
FROM joined_data
GROUP BY week, region, channel;
```

### 5. Train the Model

```bash
# Run the training notebook in Snowflake
./run.sh main
```

Or open `notebooks/01_mmm_training.ipynb` in Snowflake Notebooks and run all cells.

**Expected Output**:

```
MODEL PERFORMANCE METRICS
==========================
In-Sample R²:      0.92
CV MAPE:           12.3% ± 2.1%
CV R²:             0.87 ± 0.03

Model Quality: GOOD (CV MAPE = 12.3%)

TOP PERFORMING CHANNELS (by ROI)
--------------------------------
LINKEDIN_GLOBAL_ALL: ROI = 3.24 [2.81, 3.67] *
GOOGLE_GLOBAL_ALL: ROI = 2.15 [1.89, 2.41] *
WEBINARS_GLOBAL_ALL: ROI = 1.98 [1.45, 2.51] *
```

### 6. Deploy the Streamlit App

```bash
./deploy.sh
./run.sh streamlit
```

---

# PART 7: THE DEMO NARRATIVE

## The "Wow" Moment

**CMO asks**: *"How should we reallocate our Q3 budget to maximize industrial distributor sales in APAC?"*

**The system**:

1. **Parses the natural language question** via Cortex Analyst
2. **Queries V_MMM_RESULTS_ANALYSIS** filtered by `SEGMENT='Safety & Industrial'` and `SUPER_REGION='Asia Pacific'`
3. **Compares marginal ROI** across channels:
   - LinkedIn: mROI = 2.3x (efficient zone)
   - Google Search: mROI = 1.4x (diminishing)
   - Display: mROI = 0.8x (saturated)
4. **Runs the budget optimizer** with ±30% constraints
5. **Returns a confident answer**:

> "Shift $500k from Programmatic Display to LinkedIn Video. Predicted revenue lift: $2.4M (90% CI: $1.8M - $3.1M). Display is in saturation—each additional dollar there only returns $0.80. LinkedIn still has headroom with $2.30 return per incremental dollar."

The response is displayed alongside an **interactive response curve** showing exactly where each channel sits on its diminishing returns curve.

## Persona-Specific Value

### For the CMO

**What they see**: Executive dashboard with ROI by channel, confidence intervals, and recommended reallocations.

**What they say to the board**: "We're 90% confident LinkedIn ROI is between 2.8x and 3.6x. Shifting $500k from saturated Display to LinkedIn will generate an estimated $2.4M in incremental revenue without increasing total budget."

**Technical backing**: Bootstrap confidence intervals, IS_SIGNIFICANT flags, budget-neutral optimization.

### For the Regional Demand Lead

**What they see**: Granular results filtered to their region (e.g., "Western Europe only"), with channel-specific recommendations.

**What they use it for**: Defending regional budget requests with data. "LinkedIn in EMEA has mROI of 3.1x while APAC is at 1.8x—we should shift budget toward EMEA where there's more headroom."

**Technical backing**: Multi-dimensional granularity (Channel × Region × Product), marginal ROI at current spend levels.

### For the Data Scientist

**What they see**: Full model diagnostics—CV MAPE, R², learned adstock decay rates, saturation parameters.

**What they validate**: "CV MAPE of 12.3% means our predictions are typically within 12% of actual. The model generalizes well—not just overfitting to historical data."

**Technical backing**: Time-series cross-validation, bootstrap uncertainty quantification, interpretable parameters.

---

# PART 8: TECHNICAL SPECIFICATIONS

## Infrastructure Requirements

- **Snowflake Account**: Enterprise Edition or higher (for Snowpark ML)
- **Compute Pool**: CPU_X64_S (1 node minimum)
- **Warehouse**: XSMALL is sufficient for training
- **External Access**: PyPI integration for nevergrad installation

## Python Dependencies

```
pandas >= 2.0
numpy >= 1.24
scikit-learn >= 1.3
scipy >= 1.11
nevergrad >= 0.6
```

## Execution Time

- **Training**: ~5-10 minutes for 500 Nevergrad iterations + 100 bootstrap samples
- **Refresh Cadence**: Weekly via Snowflake Tasks or manual trigger

## Data Volume Expectations

| Table | Expected Rows |
|-------|---------------|
| GEOGRAPHY | ~90 (3 super-regions + 15 regions + 75 countries) |
| PRODUCT_CATEGORY | ~40 (4 segments + 14 divisions + 23 categories) |
| MARKETING_CHANNEL | 23 |
| MARKETING_CAMPAIGN | ~500 |
| MARKETING_MEDIA_SPEND | ~10,000-100,000 |
| V_MMM_INPUT_WEEKLY | ~2,000 (104 weeks × dimensions) |

## Model Quality Targets

| Metric | Target | Excellent |
|--------|--------|-----------|
| In-Sample R² | > 0.85 | > 0.92 |
| CV MAPE | < 20% | < 12% |
| CV R² | > 0.70 | > 0.85 |
| Significant Channels | > 60% | > 80% |

---

# GLOSSARY

**Adstock**: Mathematical transformation that spreads advertising spend across future time periods to model the "carryover effect" where ads continue influencing behavior after exposure.

**Bootstrap**: Statistical resampling technique that estimates uncertainty by repeatedly sampling from the data with replacement and re-calculating statistics.

**Confidence Interval (CI)**: Range of values that likely contains the true parameter value. A 90% CI means we're 90% confident the true value lies within the range.

**Cross-Validation (CV)**: Technique for evaluating model generalization by training on a subset of data and testing on the held-out portion.

**Diminishing Returns**: Economic principle that each additional unit of input produces less additional output than the previous unit.

**Hill Function**: S-shaped curve from pharmacology used to model saturation/diminishing returns in marketing.

**Marginal ROI (mROI)**: Return on the *next* dollar spent, calculated as the derivative of the response curve at current spend level.

**Marketing Mix Modeling (MMM)**: Statistical technique that estimates the impact of marketing activities on sales using aggregate historical data.

**Nevergrad**: Meta's open-source library for derivative-free optimization, including evolutionary algorithms.

**Ridge Regression**: Linear regression with L2 penalty that prevents overfitting by shrinking coefficients toward zero.

**ROAS**: Return on Ad Spend = Revenue attributed to ads / Ad spend.

**Saturation**: Point at which additional marketing spend produces minimal incremental response.

**Time-Series CV**: Cross-validation that respects temporal order—never training on future data to predict the past.

---

*This document provides complete context for creating podcast scripts, demo videos, or presentation materials about the Global B2B MMM & ROI Engine solution.*
