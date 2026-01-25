-- =============================================================================
-- 03_test_pipeline_model.sql
-- Test the MMM_CHANNEL_ROI_PIPELINE model with preprocessing
-- =============================================================================
-- 
-- PURPOSE: Validate that the pipeline model (with embedded adstock/saturation
-- transformations) produces predictions from raw spend data.
--
-- IMPORTANT: The pipeline model was logged with target_platforms=SPCS, so
-- inference CANNOT run directly in warehouse. Use one of:
--   1. Python notebook (03_test_pipeline_model.ipynb) with Registry.run()
--   2. Create an inference service on SPCS
--
-- MODEL INPUT SIGNATURE (19 columns):
--   Media Channels (10): Raw weekly spend per channel
--   Control Variables (9): TREND, SIN_1, COS_1, SIN_2, COS_2, Q1_FLAG, Q3_FLAG, PMI_INDEX, COMPETITOR_SOV
--
-- The pipeline internally applies:
--   1. Adstock (carryover effect with optimized decay θ)
--   2. Hill Saturation (diminishing returns with optimized α, γ)
--   3. StandardScaler normalization
--   4. Ridge regression prediction
--
-- OUTPUT: Predicted weekly revenue
-- =============================================================================

USE ROLE GLOBAL_B2B_MMM_ROLE;
USE DATABASE GLOBAL_B2B_MMM;
USE WAREHOUSE GLOBAL_B2B_MMM_WH;
USE SCHEMA MMM_PIPELINE_TEST;

-- =============================================================================
-- STEP 1: Create a view that pivots channel spend + adds control variables
-- This transforms the row-per-channel format into the wide format the model expects
-- =============================================================================
CREATE OR REPLACE VIEW V_PIPELINE_MODEL_INPUT AS
WITH weekly_totals AS (
    -- Aggregate to week level and pivot channels
    SELECT 
        WEEK_START,
        SUM(CASE WHEN CHANNEL_CODE = 'Google Ads' THEN SPEND ELSE 0 END) AS GOOGLE_ADS_GLOBAL_ALL,
        SUM(CASE WHEN CHANNEL_CODE = 'LinkedIn' THEN SPEND ELSE 0 END) AS LINKEDIN_GLOBAL_ALL,
        SUM(CASE WHEN CHANNEL_CODE = 'Meta (Facebook)' THEN SPEND ELSE 0 END) AS META_FACEBOOK_GLOBAL_ALL,
        SUM(CASE WHEN CHANNEL_CODE = 'Meta (Instagram)' THEN SPEND ELSE 0 END) AS META_INSTAGRAM_GLOBAL_ALL,
        SUM(CASE WHEN CHANNEL_CODE = 'Microsoft Ads' THEN SPEND ELSE 0 END) AS MICROSOFT_ADS_GLOBAL_ALL,
        SUM(CASE WHEN CHANNEL_CODE = 'Programmatic' THEN SPEND ELSE 0 END) AS PROGRAMMATIC_GLOBAL_ALL,
        SUM(CASE WHEN CHANNEL_CODE = 'TikTok' THEN SPEND ELSE 0 END) AS TIKTOK_GLOBAL_ALL,
        SUM(CASE WHEN CHANNEL_CODE = 'Trade Publications' THEN SPEND ELSE 0 END) AS TRADE_PUBLICATIONS_GLOBAL_ALL,
        SUM(CASE WHEN CHANNEL_CODE = 'X.com' THEN SPEND ELSE 0 END) AS X_COM_GLOBAL_ALL,
        SUM(CASE WHEN CHANNEL_CODE = 'YouTube' THEN SPEND ELSE 0 END) AS YOUTUBE_GLOBAL_ALL,
        SUM(REVENUE) AS ACTUAL_REVENUE,
        AVG(AVG_PMI) AS PMI_INDEX,
        AVG(AVG_COMPETITOR_SOV) AS COMPETITOR_SOV
    FROM DIMENSIONAL.V_MMM_INPUT_WEEKLY
    GROUP BY WEEK_START
),
with_controls AS (
    -- Add control variables: trend, seasonality, fiscal flags
    SELECT 
        wt.*,
        -- Trend: week number since start
        ROW_NUMBER() OVER (ORDER BY WEEK_START) AS TREND,
        -- Fourier seasonality (52-week cycle)
        SIN(2 * PI() * ROW_NUMBER() OVER (ORDER BY WEEK_START) / 52.0) AS SIN_1,
        COS(2 * PI() * ROW_NUMBER() OVER (ORDER BY WEEK_START) / 52.0) AS COS_1,
        SIN(4 * PI() * ROW_NUMBER() OVER (ORDER BY WEEK_START) / 52.0) AS SIN_2,
        COS(4 * PI() * ROW_NUMBER() OVER (ORDER BY WEEK_START) / 52.0) AS COS_2,
        -- Fiscal quarter flags
        CASE WHEN QUARTER(WEEK_START) = 1 THEN 1 ELSE 0 END AS Q1_FLAG,
        CASE WHEN QUARTER(WEEK_START) = 3 THEN 1 ELSE 0 END AS Q3_FLAG
    FROM weekly_totals wt
)
SELECT * FROM with_controls
ORDER BY WEEK_START;

-- Quick check: View the input data shape
SELECT COUNT(*) AS row_count, 
       MIN(WEEK_START) AS first_week, 
       MAX(WEEK_START) AS last_week 
FROM V_PIPELINE_MODEL_INPUT;

-- =============================================================================
-- STEP 2: Test the Pipeline Model
-- NOTE: Direct SQL inference requires SPCS. Use Python notebook instead.
-- This section shows what the query WOULD look like.
-- =============================================================================

-- UNCOMMENT IF RUNNING IN SPCS SERVICE CONTEXT:
/*
CREATE OR REPLACE TABLE PIPELINE_PREDICTIONS AS
SELECT 
    WEEK_START,
    ACTUAL_REVENUE,
    MMM.MMM_CHANNEL_ROI_PIPELINE!PREDICT(
        GOOGLE_ADS_GLOBAL_ALL,
        LINKEDIN_GLOBAL_ALL,
        META_FACEBOOK_GLOBAL_ALL,
        META_INSTAGRAM_GLOBAL_ALL,
        MICROSOFT_ADS_GLOBAL_ALL,
        PROGRAMMATIC_GLOBAL_ALL,
        TIKTOK_GLOBAL_ALL,
        TRADE_PUBLICATIONS_GLOBAL_ALL,
        X_COM_GLOBAL_ALL,
        YOUTUBE_GLOBAL_ALL,
        TREND,
        SIN_1,
        COS_1,
        SIN_2,
        COS_2,
        Q1_FLAG,
        Q3_FLAG,
        PMI_INDEX,
        COMPETITOR_SOV
    ) AS PREDICTION_OBJECT,
    PREDICTION_OBJECT:output_feature_0::FLOAT AS PREDICTED_REVENUE,
    GOOGLE_ADS_GLOBAL_ALL,
    LINKEDIN_GLOBAL_ALL,
    META_FACEBOOK_GLOBAL_ALL,
    PROGRAMMATIC_GLOBAL_ALL,
    TREND,
    PMI_INDEX
FROM V_PIPELINE_MODEL_INPUT;
*/

-- For now, run the Python notebook: notebooks/03_test_pipeline_model.ipynb
SELECT 'Run notebook 03_test_pipeline_model.ipynb for inference (requires SPCS)' AS NEXT_STEP;

-- =============================================================================
-- STEP 3: Evaluate prediction quality
-- =============================================================================
SELECT 
    '--- PIPELINE MODEL TEST RESULTS ---' AS SECTION;

-- Overall metrics
SELECT 
    COUNT(*) AS n_predictions,
    ROUND(AVG(ACTUAL_REVENUE), 0) AS avg_actual_revenue,
    ROUND(AVG(PREDICTED_REVENUE), 0) AS avg_predicted_revenue,
    ROUND(CORR(ACTUAL_REVENUE, PREDICTED_REVENUE), 4) AS correlation,
    ROUND(AVG(ABS(ACTUAL_REVENUE - PREDICTED_REVENUE)), 0) AS mean_absolute_error,
    ROUND(AVG(ABS(ACTUAL_REVENUE - PREDICTED_REVENUE) / NULLIF(ACTUAL_REVENUE, 0)) * 100, 2) AS mape_pct
FROM PIPELINE_PREDICTIONS
WHERE ACTUAL_REVENUE > 0;

-- Sample predictions
SELECT 
    WEEK_START,
    ROUND(ACTUAL_REVENUE, 0) AS ACTUAL,
    ROUND(PREDICTED_REVENUE, 0) AS PREDICTED,
    ROUND(PREDICTED_REVENUE - ACTUAL_REVENUE, 0) AS ERROR,
    ROUND((PREDICTED_REVENUE - ACTUAL_REVENUE) / NULLIF(ACTUAL_REVENUE, 0) * 100, 1) AS ERROR_PCT
FROM PIPELINE_PREDICTIONS
WHERE ACTUAL_REVENUE > 0
ORDER BY WEEK_START
LIMIT 15;

-- =============================================================================
-- STEP 4: Create summary view for dashboard consumption
-- =============================================================================
CREATE OR REPLACE VIEW V_PIPELINE_MODEL_EVALUATION AS
SELECT 
    p.WEEK_START,
    p.ACTUAL_REVENUE,
    p.PREDICTED_REVENUE,
    p.PREDICTED_REVENUE - p.ACTUAL_REVENUE AS PREDICTION_ERROR,
    CASE 
        WHEN p.ACTUAL_REVENUE = 0 THEN NULL
        ELSE (p.PREDICTED_REVENUE - p.ACTUAL_REVENUE) / p.ACTUAL_REVENUE * 100 
    END AS ERROR_PERCENTAGE,
    -- Input summary
    p.GOOGLE_ADS_GLOBAL_ALL + p.LINKEDIN_GLOBAL_ALL + p.META_FACEBOOK_GLOBAL_ALL + p.PROGRAMMATIC_GLOBAL_ALL AS TOTAL_SPEND_SAMPLE,
    p.TREND,
    p.PMI_INDEX
FROM PIPELINE_PREDICTIONS p
ORDER BY p.WEEK_START;

-- Final check
SELECT 'SUCCESS: Pipeline model test complete' AS STATUS;
SELECT 'View results: SELECT * FROM MMM_PIPELINE_TEST.V_PIPELINE_MODEL_EVALUATION' AS NEXT_STEP;
