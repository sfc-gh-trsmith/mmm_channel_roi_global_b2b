Demo Requirements Document (DRD): Global B2B Marketing Mix Modeling (MMM) & ROI Engine

### **1\. Strategic Overview**

**Problem Statement:** For a global diversified manufacturing conglomerate (operating across Industrial, Healthcare, and Consumer lines), marketing data is fragmented across ad platforms (Sprinklr), CRM (Salesforce), and ERP (SAP). Due to long sales cycles (6–18 months) and a heavy reliance on distributor partners, the organization cannot correlate top-of-funnel digital spend with actual booked revenue. This leads to inefficient "peanut butter" budget spreading rather than data-driven allocation.  
**Target Business Goals (KPIs):**

1. **Optimize Marketing Efficiency Ratio (MER):** Improve revenue generated per dollar of marketing spend by 15% through algorithmic budget reallocation.  
2. **Accelerate Pipeline Velocity:** Reduce average deal cycle time by identifying and funding the specific campaign types that drive faster "Lead-to-Opportunity" conversion.

**The "Wow" Moment:** The CMO asks the Cortex Analyst agent: *"How should we reallocate our Q3 budget to maximize industrial distributor sales in APAC?"* 

The system:
1. **Queries the MMM results** filtered by `SEGMENT='Safety & Industrial'` and `SUPER_REGION='Asia Pacific'`
2. **Compares marginal ROI** across channels to identify saturation (Display at 0.8x mROI) vs. headroom (LinkedIn at 2.3x mROI)
3. **Runs the budget optimizer** with ±30% constraints to find optimal reallocation
4. **Returns a confident answer**: *"Shift $500k from Programmatic Display to LinkedIn Video. Predicted revenue lift: $2.4M (90% CI: $1.8M - $3.1M). Display is in saturation—each additional dollar there only returns $0.80. LinkedIn still has headroom with $2.30 return per incremental dollar."*

The response is displayed alongside an **interactive saturation curve** showing exactly where each channel sits on its diminishing returns curve.

### **2\. User Personas & Stories**

| Persona Level | Role Title | Key User Story (Demo Flow) |
| :---- | :---- | :---- |
| **Strategic** | **CMO / VP of Global Markets** | "As a CMO, I want to see a unified view of ROI **with confidence intervals** across all business units (Safety, Healthcare, Transport) so I can defend my budget to the CFO with statistically-backed attribution—not just point estimates." |
| **Operational** | **Regional Demand Gen Lead** | "As a Demand Lead, I want to simulate 'What-If' scenarios **at my regional level**—specifically, if I shift $200k from Display to LinkedIn in Western Europe, what's the predicted revenue lift **and how confident is that estimate**?" |
| **Technical** | **Data Scientist** | "As a DS, I want to run a production-grade MMM with **proper time-series cross-validation, bootstrap uncertainty, and constrained optimization**—all within Snowpark without extracting data to external servers." |

#### **Persona-Specific Model Features**

| Persona | Key Model Output | Why It Matters |
|---------|-----------------|----------------|
| **CMO** | ROI with 90% CI (e.g., "LinkedIn ROI: 3.2x [2.8, 3.6]") | Can confidently state "LinkedIn is between 2.8x and 3.6x ROI" to the board |
| **CMO** | IS_SIGNIFICANT flag | Knows which channels have statistically reliable estimates vs. noisy data |
| **Regional Lead** | Granular results by SUPER_REGION × SEGMENT × CHANNEL | Sees that "LinkedIn works better in EMEA than APAC for Safety products" |
| **Regional Lead** | MARGINAL_ROI (not just average ROI) | Knows where the *next* dollar should go, not just historical performance |
| **Data Scientist** | CV MAPE < 15%, out-of-sample R² | Validates model generalizes to future data, not just overfitting history |
| **Data Scientist** | ADSTOCK_DECAY_RATE per channel | Can explain "LinkedIn has 0.75 decay (long consideration) vs. Search at 0.25 (immediate intent)" |

### **3\. Data Architecture & Snowpark ML (Backend)**

#### **3.1 Schema Organization**

| Schema | Purpose | Contents |
|--------|---------|----------|
| **RAW** | Landing zone | Raw data from Sprinklr, Salesforce, SAP |
| **ATOMIC** | Normalized & clean | Dimension tables, fact tables with SCD2 versioning |
| **DIMENSIONAL** | Flattened views | Pre-joined views for analytics and Streamlit |
| **MMM** | Analytics mart | Model results, legacy views |

#### **3.2 ATOMIC Schema - Core Tables**

**Dimension Tables (Hierarchical):**

| Table | Hierarchy | Purpose |
|-------|-----------|---------|
| `GEOGRAPHY` | Super-Region → Region → Country | 75+ countries in 3-tier hierarchy |
| `PRODUCT_CATEGORY` | Segment → Division → Category | 23 product lines across 4 segments |
| `ORGANIZATION` | Corporate → Business Group → Regional BU | Business unit rollup |
| `MARKETING_CHANNEL` | Flat | 23 marketing channels with benchmarks |
| `CURRENCY` | Flat | Currency reference |

**Marketing Fact Tables:**

| Table | Grain | Source | Key Columns |
|-------|-------|--------|-------------|
| `MARKETING_CAMPAIGN` | Campaign | Sprinklr | CAMPAIGN_CODE (taxonomy), FK to all dimensions |
| `MARKETING_MEDIA_SPEND` | Daily × Campaign | Sprinklr | AD_DATE, SPEND_USD, IMPRESSIONS, CLICKS |
| `MARKET_INDICATOR` | Weekly × Geography | External | PMI_INDEX, COMPETITOR_SOV |
| `MMM_MODEL_RESULT` | Model Run × Dimensions | ML Output | ROI, MARGINAL_ROI, COEFFICIENT_WEIGHT |
| `ACTUAL_FINANCIAL_RESULT` | Invoice | SAP | REVENUE_AMOUNT, POSTING_DATE |
| `OPPORTUNITY` | Opportunity | Salesforce | STAGE_NAME, AMOUNT, CLOSE_DATE |

#### **3.3 DIMENSIONAL Schema - Pre-Joined Views**

**Flattened Hierarchy Views:**

| View | Source | Output |
|------|--------|--------|
| `DIM_GEOGRAPHY_HIERARCHY` | GEOGRAPHY (self-join) | COUNTRY_NAME, REGION_NAME, SUPER_REGION_NAME |
| `DIM_PRODUCT_HIERARCHY` | PRODUCT_CATEGORY (self-join) | CATEGORY_NAME, DIVISION_NAME, SEGMENT_NAME |
| `DIM_ORGANIZATION_HIERARCHY` | ORGANIZATION (self-join) | BU_NAME, BUSINESS_GROUP_NAME |
| `DIM_CAMPAIGN` | All above + MARKETING_CHANNEL | Fully denormalized campaign dimension |

**Fact Views for Streamlit:**

| View | Purpose | Key Metrics |
|------|---------|-------------|
| `FACT_MEDIA_SPEND_DAILY` | Dashboard drill-down | SPEND_USD, CTR, CPM with all dimensions |
| `V_MMM_INPUT_WEEKLY` | Model training input | Weekly aggregated spend + revenue + controls |
| `V_MMM_RESULTS_ANALYSIS` | What-If simulator | ROI, MARGINAL_ROI with all dimensions |
| `V_FILTER_REGIONS` | Streamlit filter population | Region dropdown values |
| `V_FILTER_PRODUCTS` | Streamlit filter population | Product dropdown values |

#### **3.4 Data Flow Diagram**

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Sprinklr     │     │   Salesforce    │     │      SAP        │
│  (Ad Platform)  │     │     (CRM)       │     │     (ERP)       │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                        RAW SCHEMA                                │
│  SPRINKLR_DAILY  │  SFDC_OPPORTUNITIES  │  SAP_ACTUALS          │
└─────────────────────────────┬───────────────────────────────────┘
                              │ ETL / Transform
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ATOMIC SCHEMA                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │    GEOGRAPHY    │  │ PRODUCT_CATEGORY│  │  ORGANIZATION   │  │
│  │   (Hierarchy)   │  │   (Hierarchy)   │  │   (Hierarchy)   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │MARKETING_CHANNEL│  │MARKETING_CAMPAIGN│ │MARKETING_MEDIA_ │  │
│  │                 │  │   (Taxonomy)    │  │     SPEND       │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │MARKET_INDICATOR │  │  OPPORTUNITY    │  │ACTUAL_FINANCIAL │  │
│  │  (Controls)     │  │   (Pipeline)    │  │    _RESULT      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────┬───────────────────────────────────┘
                              │ Views (flatten hierarchies)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DIMENSIONAL SCHEMA                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │DIM_GEOGRAPHY_   │  │DIM_PRODUCT_     │  │  DIM_CAMPAIGN   │  │
│  │   HIERARCHY     │  │   HIERARCHY     │  │ (Denormalized)  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│  ┌─────────────────┐  ┌─────────────────┐                       │
│  │FACT_MEDIA_SPEND │  │V_MMM_INPUT_     │  → Streamlit App     │
│  │     _DAILY      │  │    WEEKLY       │  → Robyn/PyMC Model  │
│  └─────────────────┘  └─────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

#### **3.5 Unstructured Data (Tribal Knowledge)**

* **Source Material:** Campaign Briefs, Creative Strategy Decks (PDFs), Competitor Analysis Reports.  
* **Purpose:** Used by Cortex Search to provide context on *why* a specific campaign worked (e.g., "What was the messaging hook for the 'Science of Safety' campaign?").

#### **3.6 MMM Model Architecture & Technical Approach**

##### **Why This Modeling Approach Matters for the Demo**

The MMM implementation uses techniques pioneered by Meta's Robyn, Google's LightweightMMM, and PyMC-Marketing to deliver **credible, defensible insights** that resonate with each persona:

| Persona | What They Care About | How the Model Delivers |
|---------|---------------------|------------------------|
| **CMO** | "Can I trust this number?" | **90% confidence intervals** on ROI show reliability |
| **Regional Lead** | "What's specific to MY market?" | **Multi-dimensional granularity** (Channel × Region × Product) |
| **Data Scientist** | "Is this methodologically sound?" | **Time-series CV**, proper out-of-sample validation |

##### **Core Modeling Techniques**

| Technique | Purpose | Business Translation |
|-----------|---------|---------------------|
| **Geometric Adstock** | Models the "carryover" effect where ads influence behavior for weeks after exposure | "LinkedIn ads have a 70% decay rate—they keep working for 4-6 weeks after you stop spending" |
| **Hill Saturation** | Captures diminishing returns as spend increases | "After $50k/week on Google Search, each additional dollar generates less incremental revenue" |
| **Nevergrad Optimization** | Evolutionary algorithm finds optimal decay/saturation parameters per channel | "The model automatically learns that LinkedIn has longer carryover than Search" |
| **Ridge Regression** | Prevents overfitting with regularization; enforces positive media coefficients | "Every channel either helps or is neutral—no negative impacts from advertising" |

##### **Best Practice Enhancements (v3.0)**

| Feature | Implementation | Demo Value |
|---------|----------------|------------|
| **Time-Series Cross-Validation** | Rolling 52-week train / 13-week test windows | Proves model works on unseen future data (CV MAPE < 15%) |
| **Bootstrap Confidence Intervals** | 100 iterations, 90% CI on ROI | Shows CMO which channels have "tight" vs. "uncertain" estimates |
| **External Controls** | Fourier seasonality + PMI + Competitor SOV | Isolates true marketing impact from economic cycles |
| **Budget Optimizer** | Constrained SLSQP with ±30% limits per channel | Realistic reallocation recommendations (not "put 100% in LinkedIn") |
| **Marginal ROI** | Derivative of response curve at current spend | Answers "Where should my NEXT dollar go?" not just historical ROI |

##### **Model Output Schema**

Results are written at the **Channel × Region × Product** grain with full dimensional keys:

```
ATOMIC.MMM_MODEL_RESULT
├── MODEL_RUN_DATE, MODEL_VERSION
├── Dimensional Keys: MARKETING_CHANNEL_ID, GEOGRAPHY_ID, PRODUCT_CATEGORY_ID, ORGANIZATION_ID
├── Core Metrics: ROI, MARGINAL_ROI, COEFFICIENT_WEIGHT
├── Uncertainty: ROI_CI_LOWER, ROI_CI_UPPER, IS_SIGNIFICANT
├── Parameters: ADSTOCK_DECAY_RATE, SATURATION_POINT
└── Recommendations: OPTIMAL_SPEND_SUGGESTION
```

##### **Technical Specifications**

* **Infrastructure:** Snowflake Notebooks or Snowpark Container Services (SPCS)
* **Python Version:** 3.11 (Snowflake Anaconda channel compatible)
* **Key Libraries:** `nevergrad`, `scikit-learn`, `scipy`, `pandas`, `numpy`
* **Training Input:** `DIMENSIONAL.V_MMM_INPUT_WEEKLY` (pre-aggregated weekly data)
* **Model Output:** `ATOMIC.MMM_MODEL_RESULT` with FK joins to all dimension tables
* **Execution Time:** ~5-10 minutes for 500 Nevergrad iterations + 100 bootstrap samples
* **Refresh Cadence:** Weekly via Snowflake Tasks or manual trigger

##### **Response Curve Visualization**

The model generates response curves that power the Streamlit "What-If Simulator":

```
                    Response Curve: LinkedIn_EMEA_SI
Revenue              ┌────────────────────────────────────┐
Contribution         │                     ●●●●●●●●●●●●●  │ ← Saturation zone
    ▲                │                ●●●●●               │   (diminishing returns)
    │                │           ●●●●●                    │
    │                │       ●●●●                         │
    │                │    ●●●                             │
    │                │  ●●                                │ ← Efficient zone
    │                │●●                                  │   (highest marginal ROI)
    └────────────────┴────────────────────────────────────►
                     $0        $50k       $100k     Spend/Week
                              ▲
                     Current Spend ($45k)
                     Marginal ROI = 2.3x
                     "Next $10k generates $23k revenue"
```

##### **Demo "Wow" Moment Technical Flow**

```
CMO: "How should we reallocate Q3 budget for Industrial in APAC?"
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. Cortex Analyst parses natural language                       │
│    → Extracts: SEGMENT='Safety & Industrial', SUPER_REGION='APAC' │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Query V_MMM_RESULTS_ANALYSIS with filters                    │
│    → Returns: MARGINAL_ROI, CURRENT_SPEND, OPTIMAL_SPEND        │
│       per Channel (LinkedIn, Google, Display, etc.)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Budget Optimizer calculates reallocation                     │
│    → Constraint: Total budget unchanged, ±30% per channel       │
│    → Output: "Shift $500k from Display → LinkedIn"              │
│    → Predicted Lift: $2.4M (4.8x ROI on reallocation)          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Streamlit displays interactive response curve                │
│    → Shows saturation point for Display (already maxed out)     │
│    → Shows headroom for LinkedIn (still in efficient zone)      │
│    → 90% CI shows this is a "confident" recommendation          │
└─────────────────────────────────────────────────────────────────┘
```

### **4\. Cortex Intelligence Specifications**

**Cortex Analyst (Structured Data / SQL)**

* **Semantic Model Scope:**  
  * **Measures:** Total Spend, Return on Ad Spend (ROAS), Marginal ROI, Incremental Revenue, CTR, CPM, CPC  
  * **Dimensions:** 
    - Geography: Super-Region → Region → Country
    - Product: Segment → Division → Category (23 product lines)
    - Channel: Channel Type → Channel
    - Time: Year → Quarter → Month → Week
    - Organization: Business Group → Regional BU

* **Golden Queries (Verification):**

| User Prompt | Expected View | Expected SQL Pattern |
|-------------|---------------|---------------------|
| "Show me ROAS by business segment for APAC" | `V_MMM_INPUT_WEEKLY` | `SELECT SEGMENT, SUM(REVENUE)/SUM(SPEND) AS ROAS FROM V_MMM_INPUT_WEEKLY WHERE SUPER_REGION = 'Asia Pacific' GROUP BY SEGMENT` |
| "Compare LinkedIn vs Google ROAS for Abrasives in Germany" | `V_MMM_INPUT_WEEKLY` | `SELECT CHANNEL, SUM(REVENUE)/SUM(SPEND) AS ROAS FROM V_MMM_INPUT_WEEKLY WHERE CATEGORY = 'Abrasives' AND COUNTRY = 'Germany' AND CHANNEL IN ('LINKEDIN', 'GOOGLE_SEARCH') GROUP BY CHANNEL` |
| "What's the marginal ROI by channel for PPE in Western Europe?" | `V_MMM_RESULTS_ANALYSIS` | `SELECT CHANNEL_NAME, MARGINAL_ROI FROM V_MMM_RESULTS_ANALYSIS WHERE CATEGORY_NAME = 'Personal Protective Equipment' AND REGION_NAME = 'Western Europe'` |
| "Show spend trend by product division in Greater China" | `FACT_MEDIA_SPEND_DAILY` | `SELECT AD_MONTH, DIVISION_NAME, SUM(SPEND_USD) FROM FACT_MEDIA_SPEND_DAILY WHERE REGION_NAME = 'Greater China' GROUP BY 1, 2 ORDER BY 1` |

**Cortex Search (Unstructured Data / RAG)**

* **Service Name:** MARKETING\_KNOWLEDGE\_BASE  
* **Indexing Strategy:**  
  * **Document Attribute:** Index by Campaign\_ID, Product Category, and Region to link strategy docs to performance metrics.  
* **Sample RAG Prompts:**
  - "Summarize the creative strategy for our top-performing LinkedIn campaign in Q2."
  - "What messaging worked best for PPE campaigns in APAC?"
  - "How did the 'Science of Safety' campaign perform vs. objectives?"

### **5\. Streamlit Application UX/UI**

**Layout Strategy:**

* **Page 1 (Executive - The Allocator):** High-level KPI cards (Total Spend, Blended ROAS). Central feature is the **Budget Optimizer**: A "Before/After" donut chart showing current allocation vs. AI-recommended allocation to maximize revenue.
  * **Data Source:** `DIMENSIONAL.V_MMM_RESULTS_ANALYSIS`

* **Page 2 (Tactical - The Simulator):** A "Flight Simulator" interface. Sliders allow the user to adjust spend per channel (+/- %) and see a real-time predicted impact curve on Revenue, generated by querying the model results.
  * **Data Source:** `DIMENSIONAL.V_MMM_RESULTS_ANALYSIS` (marginal ROI curves)

* **Page 3 (Regional Heatmap):** Interactive world map showing ROI performance by country with drill-down capability. Click a country to see channel breakdown.
  * **Data Source:** `DIMENSIONAL.FACT_MEDIA_SPEND_DAILY` aggregated by `COUNTRY_NAME`
  * **Visualization:** Plotly choropleth map with ISO country codes

* **Page 4 (Product Portfolio):** Matrix view of all 23 product categories × regions showing performance metrics. Heatmap colors indicate ROI strength.
  * **Data Source:** `DIMENSIONAL.V_MMM_INPUT_WEEKLY` pivoted by CATEGORY × SUPER_REGION
  * **Visualization:** Altair heatmap with click-to-drill

**Filter Components (All Pages):**

| Filter | Source View | Cascade Behavior |
|--------|-------------|------------------|
| Super-Region | `V_FILTER_REGIONS` | Filters Region dropdown |
| Region | `V_FILTER_REGIONS` | Filters Country dropdown |
| Country | `DIM_GEOGRAPHY_HIERARCHY` | Independent |
| Segment | `V_FILTER_PRODUCTS` | Filters Division dropdown |
| Division | `V_FILTER_PRODUCTS` | Filters Category dropdown |
| Category | `V_FILTER_PRODUCTS` | Independent |
| Channel | `V_FILTER_CHANNELS` | Independent |
| Date Range | `V_FILTER_DATE_RANGE` | Independent |

**Component Logic:**

* **Visualizations:**  
  * **Altair Response Curves:** Showing "Diminishing Returns" (Saturation) for each channel.  
  * **Waterfall Chart:** Visualizing the "Halo Effect" of Brand Spend on Product Sales.
  * **Plotly Choropleth:** Global ROI heatmap by country.
  * **Treemap:** Spend allocation by Segment → Division → Category.
  
* **Chat Integration:** A Cortex Agent sidebar that routes questions.  
  * *Route A:* "How much did we spend in Germany on PPE?" → **Cortex Analyst** (SQL via `V_MMM_INPUT_WEEKLY`).  
  * *Route B:* "What was the tone of the Q3 ads?" → **Cortex Search** (Vector).  
  * *Route C:* "Run a new optimization for APAC." → **Snowpark ML Job** trigger.

### **6\. Success Criteria**

#### **Technical Validators**

| Metric | Target | Validation Query |
|--------|--------|------------------|
| **CV MAPE** | < 15% | Model can predict revenue within 15% on holdout quarters |
| **In-Sample R²** | > 0.85 | Model explains 85%+ of revenue variance |
| **Significant Channels** | > 60% | Majority of channels have CI that doesn't include zero |
| **Execution Time** | < 10 min | Full training completes in Snowflake Notebook |

#### **Business Validators**

| Criterion | Evidence | Demo Proof Point |
|-----------|----------|-----------------|
| **Unified Attribution** | Single ROI per Channel × Region × Product | CMO can compare "LinkedIn EMEA Safety" vs. "Google APAC Healthcare" |
| **Defensible Recommendations** | 90% confidence intervals on all ROI estimates | "We're 90% confident LinkedIn ROI is between 2.8x and 3.6x" |
| **Actionable Optimization** | Budget Optimizer with ±30% constraints | "Shift $500k from Display → LinkedIn for projected $2.4M lift" |
| **Regional Granularity** | Results at Super-Region, Region, or Country level | Regional leads can filter to "Western Europe only" and get specific guidance |
| **Product Line Insight** | Results by Segment, Division, or Category | "PPE campaigns have higher ROI than Adhesives on LinkedIn" |

#### **Demo "Wow" Moment Checklist**

- [ ] CMO asks natural language question about budget reallocation
- [ ] Cortex Analyst returns specific channel recommendations
- [ ] Response curves show diminishing returns visually
- [ ] Confidence intervals demonstrate statistical rigor
- [ ] Predicted revenue lift quantifies the business impact
- [ ] Regional drill-down shows market-specific insights


## **Data Generation Spec**

This specification outlines the requirements for generating a synthetic dataset that mimics the complexity of a global multi-industrial conglomerate.

To ensure the MMM (Marketing Mix Modeling) demo works, this data **cannot be random**. It must contain "injected signals"—mathematical correlations between Spend (Sprinklr), Pipeline (Salesforce), and Revenue (SAP) with specific **time lags** (e.g., ad spend today = revenue 9 months from now).

### 1. The "Golden Thread" Taxonomy

The success of this dataset relies on a strict naming convention and dimensional foreign keys that exist across all systems.

**Enhanced Campaign Code Pattern:** `[Segment]_[SuperRegion]_[Country]_[Division]_[Objective]_[ID]`

**Example:** `SI_EMEA_DEU_ABR_LEADGEN_CMP-2451`

| Component | Values | Reference Table |
|-----------|--------|-----------------|
| **Segment** | SI (Safety & Industrial), TE (Transportation & Electronics), HC (Healthcare), CN (Consumer) | `PRODUCT_CATEGORY` Level 1 |
| **SuperRegion** | AMER, EMEA, APAC | `GEOGRAPHY` Super-Region |
| **Country** | ISO 3-letter codes (DEU, USA, CHN, etc.) | `GEOGRAPHY` Country |
| **Division** | ABR (Abrasives), ADH (Adhesives), SAF (Safety), etc. | `PRODUCT_CATEGORY` Level 2 |
| **Objective** | BRAND, LEADGEN, NURTURE | Campaign objective |
| **ID** | CMP-XXXX | Unique identifier |

#### Product Hierarchy (23 Categories)

| Level 1 (Segment) | Level 2 (Division) | Level 3 (Category) |
|-------------------|-------------------|-------------------|
| **Safety & Industrial** | Abrasives & Surface | Abrasives, Compounds & Polishes, Coatings |
| | Adhesives & Tapes | Adhesives/Sealants/Fillers, Tapes, Films & Sheeting |
| | Safety Solutions | Personal Protective Equipment, Filtration & Separation |
| | Industrial Products | Tools & Equipment, Lubricants, Cleaning Supplies |
| | Building & Construction | Building Materials, Insulation, Signage & Marking |
| **Transportation & Electronics** | Automotive | Automotive Parts & Hardware, Advanced Materials |
| | Electronics | Electronics Materials & Components, Electrical |
| | Display & Graphics | Labels |
| **Healthcare** | Medical Solutions | Medical, Lab Supplies & Testing |
| **Consumer** | Consumer Home | Home, Office Supplies |

#### Geography Hierarchy (75+ Countries)

| Super-Region | Region | Sample Countries |
|--------------|--------|-----------------|
| **Americas** | North America | USA, Canada, Mexico |
| | Latin America North | Guatemala, Costa Rica, Panama, Colombia |
| | Latin America South | Brazil, Argentina, Chile, Peru |
| **EMEA** | Western Europe | Germany, France, UK, Italy, Spain, Netherlands |
| | Central & Eastern Europe | Poland, Czech Republic, Hungary, Romania, Russia, Turkey |
| | Middle East | UAE, Saudi Arabia, Israel, Qatar |
| | Africa | South Africa, Nigeria, Kenya, Egypt |
| **Asia Pacific** | Greater China | China, Hong Kong, Taiwan |
| | Japan & Korea | Japan, South Korea |
| | ASEAN | Singapore, Thailand, Vietnam, Indonesia, Malaysia, Philippines |
| | ANZ | Australia, New Zealand |
| | South Asia | India, Bangladesh, Pakistan |

#### Marketing Channels

| Channel Type | Channels |
|--------------|----------|
| **Social** | LinkedIn, LinkedIn Video, LinkedIn InMail, Facebook, Instagram |
| **Search** | Google Search, Bing Search |
| **Programmatic** | Google Display, Programmatic Display |
| **Video** | YouTube, Programmatic Video, Connected TV |
| **Direct** | Email, Direct Mail, Trade Publications |
| **Events** | Webinars, Trade Shows |
| **Other** | Content Syndication, Podcast, Print, Out of Home |

---

### 2. Data Source Specifications

#### Source A: Sprinklr (Ad Tech / Media Spend)

**Format:** CSV / Parquet
**Grain:** Daily per Creative
**Volume:** ~2 Years of history (needed for MMM seasonality).

**Schema Definition:**
| Column Name | Data Type | Generation Logic / "Look & Feel" |
| :--- | :--- | :--- |
| `AD_DATE` | Date | Continuous daily feed. |
| `CAMPAIGN_NAME` | Varchar | **Crucial:** Must follow Taxonomy. Ex: `SIBG_NA_ASD_FallSafety_Awareness_CMP-101` |
| `CAMPAIGN_ID` | Varchar | Unique Key (e.g., `CMP-101`). |
| `PLATFORM` | Varchar | Mix: `LinkedIn` (40%), `Google Ads` (30%), `Programmatic` (20%), `Facebook` (10%). |
| `ASSET_TYPE` | Varchar | `Video`, `Static Image`, `Carousel`, `Whitepaper`. |
| `SPEND_USD` | Float | **Signal:** Heavy spend in Q1/Q3. LinkedIn CPM ~$50, FB CPM ~$10. |
| `IMPRESSIONS` | Integer | Derived: `Spend / CPM * 1000`. |
| `CLICKS` | Integer | Derived: `Impressions * CTR` (LinkedIn CTR ~0.5%, Google ~2.0%). |
| `VIDEO_VIEWS_50`| Integer | Derived: High for Video assets, 0 for static. |
| `ENGAGEMENTS` | Integer | Shares + Comments. Use this for "Organic" signal. |

**Signal Injection Rule:**

* For **LinkedIn** campaigns targeting **Health Care**, inject a 20% increase in spend 6 months prior to any revenue spikes in the SAP data.

---

#### Source B: Salesforce (CRM / Pipeline)

**Format:** CSV / Parquet
**Grain:** Opportunity Level (Aggregated to Weekly for MMM)
**Context:** Tracks the "middle" of the funnel.

**Schema Definition:**
| Column Name | Data Type | Generation Logic / "Look & Feel" |
| :--- | :--- | :--- |
| `OPP_ID` | Varchar | `OPP-882910` |
| `ACCOUNT_NAME` | Varchar | Use manufacturing names: "General Motors," "Mayo Clinic," "Foxconn." |
| `CREATED_DATE` | Date | **Lag Rule:** Should appear 2-6 weeks *after* a Sprinklr ad spike. |
| `CLOSE_DATE` | Date | **Cycle Rule:** `Created_Date` + 180 to 270 days (Long B2B cycle). |
| `STAGE` | Varchar | `Prospecting`, `Qualification`, `Proposal`, `Negotiation`, `Closed Won`, `Closed Lost`. |
| `AMOUNT_USD` | Float | High variance: $50k (Transactional) to $5M (Enterprise Contracts). |
| `LEAD_SOURCE` | Varchar | `Web`, `Trade Show`, `Referral`, `Partner`. |
| `CAMPAIGN_ID` | Varchar | **The Join Key:** Matches Sprinklr `CAMPAIGN_ID`. Populate this for ~40% of records (Attribution is never 100%). |
| `BUSINESS_GROUP`| Varchar | `SIBG`, `HCBG`, `TEBG`. |

**Signal Injection Rule:**

* Ensure that opportunities linked to `LinkedIn` campaigns have a 25% higher "Win Rate" (Stage = Closed Won) than opportunities linked to `Display` campaigns.

---

#### Source C: SAP (ERP / Invoiced Revenue)

**Format:** CSV / Parquet
**Grain:** Line Item per Invoice
**Context:** The "Ground Truth." Includes both Direct Sales and Channel/Distributor Sales.

**Schema Definition:**
| Column Name | Data Type | Generation Logic / "Look & Feel" |
| :--- | :--- | :--- |
| `INVOICE_NUM` | Varchar | `INV-99201` |
| `POSTING_DATE` | Date | When revenue is recognized. |
| `SOLD_TO_PARTY` | Varchar | **Distributors:** "Grainger," "Fastenal," "McKesson," "Cardinal Health." |
| `SHIP_TO_PARTY` | Varchar | The end customer (optional, often null in distribution). |
| `MATERIAL_NUM` | Varchar | `70-0064-1234-5` (Mimic standard industrial SKU formats). |
| `MATERIAL_DESC` | Varchar | "N95 Particulate Respirator," "Industrial Structural Adhesive," "Surgical Drape." |
| `PROFIT_CENTER` | Varchar | Maps to Business Group (e.g., `PC_SIBG_NA`). |
| `NET_VALUE_USD` | Float | Revenue amount. |
| `QUANTITY` | Integer | Units sold. |

**Signal Injection Rule:**

* **The Distributor Lag:** Create a correlation where high `Sprinklr Spend` in Region A results in increased `Net_Value` for Distributors in Region A roughly **9 months later**.

---

#### Source D: Contextual / Macro Data (The "Controls")

**Format:** CSV
**Grain:** Weekly

**Schema Definition:**
| Column Name | Data Type | Logic |
| :--- | :--- | :--- |
| `WEEK_START` | Date | Weekly. |
| `REGION` | Varchar | NA, EMEA, APAC. |
| `PMI_INDEX` | Float | **Purchasing Managers' Index.** Strong predictor of industrial sales. Vary between 45 (contraction) and 60 (expansion). |
| `COMPETITOR_SOV`| Float | Share of Voice metric. If Competitor SOV goes up, our sales should dip slightly. |

---

### 3. Data Generation Strategy (Python/Faker Snippet)

Use this logic to establish the causal relationships between the tables.

```python
import pandas as pd
import numpy as np
from datetime import timedelta

# 1. GENERATE CAMPAIGNS (The Anchor)
campaigns = [
    {"id": "CMP-001", "bg": "SIBG", "channel": "LinkedIn", "start": "2023-01-01", "power": 1.5}, # High ROI
    {"id": "CMP-002", "bg": "SIBG", "channel": "Display",  "start": "2023-01-01", "power": 0.2}, # Low ROI
]

# 2. GENERATE SPEND (Sprinklr)
# Daily spend with weekly seasonality
def generate_spend(campaign):
    # Logic to create daily rows for 90 days
    # Add random noise
    pass

# 3. GENERATE REVENUE (SAP) - THE "GHOST" EFFECT
# Revenue isn't random; it's a delayed echo of spend + baseline
def generate_revenue(spend_df, lag_days=270):
    revenue_rows = []
    for index, row in spend_df.iterrows():
        # Only 2% of impressions turn into revenue
        if np.random.random() < 0.02: 
            revenue_date = row['date'] + timedelta(days=lag_days + np.random.randint(-30, 30))
            
            # ROI Multiplier based on Channel Power
            deal_size = row['spend'] * row['power_multiplier'] * np.random.uniform(5, 50)
            
            revenue_rows.append({
                "posting_date": revenue_date,
                "net_value_usd": deal_size,
                "business_group": row['bg']
            })
    return pd.DataFrame(revenue_rows)

```

### 4. Implementation Steps for the Demo Team

#### Step 1: Deploy Schema Objects
```bash
# Run SQL scripts in order
snow sql -f sql/01_account_setup.sql
snow sql -f sql/02_schema_setup.sql      # Creates ATOMIC tables
snow sql -f sql/03_dimensional_views.sql # Creates DIMENSIONAL views
```

#### Step 2: Load Reference Data
```sql
-- Load reference data from CSV files
COPY INTO ATOMIC.GEOGRAPHY FROM @DATA_STAGE/ref_geography.csv FILE_FORMAT = CSV_FORMAT;
COPY INTO ATOMIC.PRODUCT_CATEGORY FROM @DATA_STAGE/ref_product_category.csv FILE_FORMAT = CSV_FORMAT;
COPY INTO ATOMIC.MARKETING_CHANNEL FROM @DATA_STAGE/ref_marketing_channel.csv FILE_FORMAT = CSV_FORMAT;
```

Reference data files:
- `data/ref_geography.csv` - 75+ countries in 3-tier hierarchy
- `data/ref_product_category.csv` - 23 product categories in 3-level hierarchy
- `data/ref_marketing_channel.csv` - 23 marketing channels with benchmark CPM/CTR

#### Step 3: Generate Synthetic Data
1. **Run Generation Script:** Create synthetic data with injected signals
   - 10,000 rows of `MARKETING_MEDIA_SPEND` (daily spend)
   - 2,000 rows of `OPPORTUNITY` (pipeline)
   - 5,000 rows of `ACTUAL_FINANCIAL_RESULT` (revenue)
   - 500 rows of `MARKETING_CAMPAIGN` (campaigns with taxonomy codes)

2. **Verify Lags:** Plot Spend vs. Revenue on a timeline. You should see Spend peaks followed by Revenue peaks ~6-9 months later. **This visual confirmation is required before proceeding.**

#### Step 4: Validate Views
```sql
-- Verify dimensional views are populated
SELECT COUNT(*) FROM DIMENSIONAL.DIM_GEOGRAPHY_HIERARCHY;    -- Should show 75+ countries
SELECT COUNT(*) FROM DIMENSIONAL.DIM_PRODUCT_HIERARCHY;      -- Should show 23 categories
SELECT COUNT(*) FROM DIMENSIONAL.V_MMM_INPUT_WEEKLY;         -- Should show weekly aggregates
```

#### Step 5: Train MMM Model

**Notebook:** `notebooks/01_mmm_training.ipynb`

**Configuration Options:**
```python
config = MMMConfig(
    geo_level="SUPER_REGION",      # or "REGION", "COUNTRY"
    product_level="SEGMENT",       # or "DIVISION", "CATEGORY"
    nevergrad_budget=500,          # Hyperparameter optimization iterations
    n_bootstrap=100,               # Bootstrap iterations for CI
    confidence_level=0.90,         # 90% confidence intervals
    cv_train_weeks=52,             # 1-year training window
    cv_test_weeks=13,              # 1-quarter holdout
    budget_change_limit=0.30       # ±30% reallocation constraint
)
```

**Execution Steps:**
1. Open notebook in Snowflake Notebooks or run via SPCS
2. Verify data loads from `DIMENSIONAL.V_MMM_INPUT_WEEKLY`
3. Monitor optimization progress (500 iterations, ~3-5 min)
4. Review CV metrics: target **MAPE < 15%**, **R² > 0.85**
5. Confirm results written to `ATOMIC.MMM_MODEL_RESULT`
6. Verify dimensional joins work in `DIMENSIONAL.V_MMM_RESULTS_ANALYSIS`

**Expected Output:**
```
MODEL PERFORMANCE METRICS
==========================
In-Sample R²:      0.92
CV MAPE:           12.3% ± 2.1%
CV R²:             0.87 ± 0.03

Model Quality: GOOD (CV MAPE = 12.3%)
```

#### Target Tables After Setup

| Schema | Table/View | Expected Rows |
|--------|------------|---------------|
| ATOMIC | GEOGRAPHY | ~90 (3 super-regions + 15 regions + 75 countries) |
| ATOMIC | PRODUCT_CATEGORY | ~40 (4 segments + 14 divisions + 23 categories) |
| ATOMIC | MARKETING_CHANNEL | 23 |
| ATOMIC | MARKETING_CAMPAIGN | ~500 |
| ATOMIC | MARKETING_MEDIA_SPEND | ~10,000 |
| ATOMIC | OPPORTUNITY | ~2,000 |
| ATOMIC | ACTUAL_FINANCIAL_RESULT | ~5,000 |
| DIMENSIONAL | V_MMM_INPUT_WEEKLY | ~2,000 (104 weeks × dimensions) |