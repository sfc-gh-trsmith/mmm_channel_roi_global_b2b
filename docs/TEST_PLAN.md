# Global B2B MMM - Full Test Cycle Plan

This document defines the complete test procedures for the Global B2B Marketing Mix Modeling project, following the principles in [SNOWFLAKE_DEMO_FULL_TEST_CYCLE.md](../.cursor/SNOWFLAKE_DEMO_FULL_TEST_CYCLE.md).

**Last Updated**: December 2025  
**Project**: Global B2B MMM Channel ROI Analysis

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Prerequisites](#2-prerequisites)
3. [Resource Naming](#3-resource-naming)
4. [Full Test Cycle](#4-full-test-cycle)
5. [Component Testing](#5-component-testing)
6. [Verification Steps](#6-verification-steps)
7. [Automated Verification Script](#7-automated-verification-script)
8. [Troubleshooting](#8-troubleshooting)
9. [CI/CD Integration](#9-cicd-integration)
10. [Known Limitations](#10-known-limitations)

---

## 1. Quick Start

### One-liner Full Test Cycle

```bash
./clean.sh --force && ./deploy.sh && ./run.sh main
```

### Step-by-Step

```bash
# 1. Clean: Remove all existing resources
./clean.sh --force -c demo

# 2. Deploy: Create infrastructure, load data, deploy apps
./deploy.sh -c demo

# 3. Run: Execute the MMM training notebook
./run.sh -c demo main

# 4. Verify: Check status and app accessibility
./run.sh -c demo status
./run.sh -c demo streamlit
```

### Expected Timeline

| Step | Duration | Notes |
|------|----------|-------|
| `clean.sh` | 15-30 sec | Fast; drops resources |
| `deploy.sh` Step 1 | 5 sec | Prerequisites check |
| `deploy.sh` Step 2 | 30-60 sec | Role, DB, WH, Compute Pool |
| `deploy.sh` Step 3 | 30-60 sec | Tables, Views, Stages |
| `deploy.sh` Step 4-5 | 1-3 min | Upload & COPY INTO |
| `deploy.sh` Step 6 | 1-2 min | Notebook deployment |
| `deploy.sh` Step 7 | 30-60 sec | Streamlit app |
| `run.sh main` | 3-10 min | Model training |

**Total**: ~8-20 minutes

---

## 2. Prerequisites

Before executing any tests:

### 2.0 No Sandbox Execution

> **Important**: These scripts **cannot run in a sandboxed environment** (e.g., Cursor AI sandbox mode).
> 
> The Snowflake CLI requires write access to its cache directory at `~/Library/Caches/pyapp/locks/` (macOS) or equivalent system paths. Sandboxed execution will fail with:
> ```
> Error: unable to open lock file /Users/.../Library/Caches/pyapp/locks/...
> Caused by: Operation not permitted (os error 1)
> ```
> 
> **Solution**: When running via Cursor AI, use `required_permissions: ["all"]` to disable the sandbox.

### 2.1 Snowflake CLI

```bash
# Verify installation
snow --version

# Test connection
snow connection test -c demo
```

### 2.2 Snowflake Connection

Connection configured in `~/.snowflake/config.toml`:

```toml
[connections.demo]
account = "your_account"
user = "your_user"
authenticator = "externalbrowser"  # or keypair
```

### 2.3 Required Permissions

- **ACCOUNTADMIN** role (for Compute Pool and External Access Integration)

### 2.4 Local Data Files

```bash
# Check synthetic data exists
ls data/synthetic/

# If missing, generate:
python utils/generate_synthetic_data.py
```

---

## 3. Resource Naming

All resources use the prefix `GLOBAL_B2B_MMM`:

### 3.1 Account-Level Resources

| Resource | Name |
|----------|------|
| Database | `GLOBAL_B2B_MMM` |
| Role | `GLOBAL_B2B_MMM_ROLE` |
| Warehouse | `GLOBAL_B2B_MMM_WH` |
| Compute Pool | `GLOBAL_B2B_MMM_COMPUTE_POOL` |
| Network Rule | `PYPI_NETWORK_RULE` |
| External Access | `PYPI_ACCESS_INTEGRATION` |

### 3.2 Schema Structure

| Schema | Purpose |
|--------|---------|
| `PUBLIC` | Network rules and shared objects |
| `RAW` | Landing zone for source data |
| `ATOMIC` | Flat staging tables |
| `DIMENSIONAL` | Analytics views |
| `MMM` | Model results, notebook, Streamlit app |

### 3.3 Key Tables

| Schema | Table | Description |
|--------|-------|-------------|
| `RAW` | `SPRINKLR_DAILY` | Raw media spend data |
| `RAW` | `SFDC_OPPORTUNITIES` | Raw opportunity data |
| `RAW` | `SAP_ACTUALS` | Raw revenue data |
| `RAW` | `MACRO_INDICATORS` | Economic indicators |
| `RAW` | `RAW_CAMPAIGN_METADATA` | Campaign metadata |
| `ATOMIC` | `MEDIA_SPEND_DAILY` | Cleaned daily spend |
| `ATOMIC` | `MARKETING_CAMPAIGN_FLAT` | Campaign metadata |
| `ATOMIC` | `OPPORTUNITY` | Salesforce opportunities |
| `ATOMIC` | `ACTUAL_FINANCIAL_RESULT` | SAP revenue |
| `ATOMIC` | `MARKET_SIGNAL` | PMI/SOV indicators |
| `MMM` | `MODEL_RESULTS` | Channel ROI with CI |
| `MMM` | `RESPONSE_CURVES` | Spend vs revenue curves |
| `MMM` | `MODEL_METADATA` | Model configuration |

---

## 4. Full Test Cycle

### 4.1 Philosophy

- **Clean Slate**: Always start from scratch for reproducibility
- **Idempotent**: Scripts work whether fresh or redeploying
- **Fail Fast**: Errors exit immediately with clear messages

### 4.2 Scenario A: Clean Slate (The "Golden Path")

**Objective**: Verify complete system from scratch.  
**Frequency**: Before every major merge/release.

```bash
./clean.sh --force && ./deploy.sh && ./run.sh main
```

### 4.3 Scenario B: Component-Level Iteration

For faster iteration during development:

| Component | Command |
|-----------|---------|
| SQL only | `./deploy.sh --only-sql` |
| Data only | `./deploy.sh --only-data` |
| Notebook only | `./deploy.sh --only-notebook && ./run.sh main` |
| Streamlit only | `./deploy.sh --only-streamlit && ./run.sh streamlit` |

### 4.4 Scenario C: Environment Prefix

Deploy to isolated environment:

```bash
./clean.sh --force -p DEV
./deploy.sh -p DEV
./run.sh -p DEV main
```

> **Limitation**: External access integration uses hardcoded database path.

---

## 5. Component Testing

### 5.1 Notebook Testing

The MMM training notebook (`notebooks/01_mmm_training.ipynb`) has specific requirements:

**Deployment Checklist**:
- [ ] Compute pool in `ACTIVE` or `IDLE` state
- [ ] `PYPI_ACCESS_INTEGRATION` exists and attached
- [ ] Live version committed

**Verification**:
```bash
# Check notebook exists
snow sql -c demo -q "SHOW NOTEBOOKS IN SCHEMA GLOBAL_B2B_MMM.MMM;"

# Check live version
snow sql -c demo -q "DESCRIBE NOTEBOOK GLOBAL_B2B_MMM.MMM.MMM_TRAINING_NOTEBOOK;"
```

**Cell Naming** (per [SNOWFLAKE_NOTEBOOK_GUIDELINES.md](../.cursor/SNOWFLAKE_NOTEBOOK_GUIDELINES.md)):
All 16 cells have meaningful metadata names:
- `mmm_overview_header`
- `install_packages`
- `imports_and_config`
- `load_data_from_snowflake`
- ... (see notebook for full list)

### 5.2 Streamlit Testing

**Deployment**:
```bash
./deploy.sh --only-streamlit
./run.sh streamlit
```

**Manual Verification**:
- [ ] Landing page loads with persona cards
- [ ] "Strategic Dashboard" shows KPIs
- [ ] "Simulator" allows budget adjustments
- [ ] "Model Explorer" shows response curves
- [ ] Charts render with channel data

### 5.3 Data Loading Testing

**Stage Verification**:
```sql
LIST @GLOBAL_B2B_MMM.ATOMIC.DATA_STAGE;
-- Expected: 5 CSV files + 75 PDF campaign briefs
```

**Row Count Verification**:
```sql
SELECT 'SPRINKLR_DAILY' as TBL, COUNT(*) as CNT FROM RAW.SPRINKLR_DAILY
UNION ALL SELECT 'SFDC_OPPORTUNITIES', COUNT(*) FROM RAW.SFDC_OPPORTUNITIES
UNION ALL SELECT 'SAP_ACTUALS', COUNT(*) FROM RAW.SAP_ACTUALS
UNION ALL SELECT 'MACRO_INDICATORS', COUNT(*) FROM RAW.MACRO_INDICATORS
UNION ALL SELECT 'RAW_CAMPAIGN_METADATA', COUNT(*) FROM RAW.RAW_CAMPAIGN_METADATA;
```

| Table | Expected Rows |
|-------|---------------|
| SPRINKLR_DAILY | ~4,300+ |
| SFDC_OPPORTUNITIES | ~17,000+ |
| SAP_ACTUALS | ~9,800+ |
| MACRO_INDICATORS | ~150+ |
| RAW_CAMPAIGN_METADATA | ~75 |

---

## 6. Verification Steps

### 6.1 Infrastructure Validation

```sql
USE ROLE ACCOUNTADMIN;

-- Compute Pool (should be IDLE, ACTIVE, or STARTING)
SHOW COMPUTE POOLS LIKE 'GLOBAL_B2B_MMM_COMPUTE_POOL';

-- Warehouse
USE ROLE GLOBAL_B2B_MMM_ROLE;
SHOW WAREHOUSES LIKE 'GLOBAL_B2B_MMM_WH';

-- Schemas
SHOW SCHEMAS IN DATABASE GLOBAL_B2B_MMM;
-- Expected: PUBLIC, RAW, ATOMIC, DIMENSIONAL, MMM

-- External Access
SHOW EXTERNAL ACCESS INTEGRATIONS LIKE 'PYPI_ACCESS_INTEGRATION';
```

### 6.2 Model Output Validation

After `./run.sh main`:

```sql
USE DATABASE GLOBAL_B2B_MMM;
USE SCHEMA MMM;

-- Check model results (should be 15 channel-region combos)
SELECT COUNT(*) as RESULT_COUNT FROM MODEL_RESULTS;

-- Check response curves (should be 1500 = 15 channels × 100 points)
SELECT COUNT(*) as CURVE_COUNT FROM RESPONSE_CURVES;

-- Check metadata (should be 1 row)
SELECT * FROM MODEL_METADATA;

-- View channel ROI summary
SELECT 
    CHANNEL,
    ROUND(ROI, 2) AS ROI,
    ROUND(MARGINAL_ROI, 2) AS MROI,
    ROUND(CURRENT_SPEND / 1000000, 1) AS SPEND_M
FROM MODEL_RESULTS
ORDER BY ROI DESC;
```

**Expected Output**:
- 15 channel-region combinations (4 channels × 4 regions, minus 1)
- Top performers: LinkedIn_APAC (~12x ROI), Google Ads_NA (~5x ROI)
- Model R² > 0.2, CV MAPE < 40%

### 6.3 View Validation

```sql
-- Check dimensional view
SELECT COUNT(*) FROM DIMENSIONAL.V_MMM_INPUT_WEEKLY;

-- Check ROI summary view
SELECT * FROM MMM.V_ROI_BY_CHANNEL;
-- Expected: 4 channels with spend and attributed revenue
```

---

## 7. Automated Verification Script

Create `verify_deployment.sh`:

```bash
#!/bin/bash
###############################################################################
# verify_deployment.sh - Automated verification for Global B2B MMM
###############################################################################

set -e
CONNECTION="${1:-demo}"
DATABASE="GLOBAL_B2B_MMM"

echo "=== Global B2B MMM Verification ==="
echo "Connection: $CONNECTION"
echo "Timestamp: $(date)"
echo ""

# Check infrastructure
echo "--- Infrastructure ---"
snow sql -c $CONNECTION -q "
    USE ROLE ACCOUNTADMIN;
    SHOW COMPUTE POOLS LIKE '${DATABASE}_COMPUTE_POOL';
" 2>&1 | head -10

# Check row counts
echo ""
echo "--- Data Row Counts ---"
snow sql -c $CONNECTION -q "
    USE ROLE ${DATABASE}_ROLE;
    USE DATABASE ${DATABASE};
    SELECT 'RAW.SPRINKLR_DAILY' as TBL, COUNT(*) as CNT FROM RAW.SPRINKLR_DAILY
    UNION ALL SELECT 'ATOMIC.MEDIA_SPEND_DAILY', COUNT(*) FROM ATOMIC.MEDIA_SPEND_DAILY
    UNION ALL SELECT 'ATOMIC.OPPORTUNITY', COUNT(*) FROM ATOMIC.OPPORTUNITY
    UNION ALL SELECT 'DIMENSIONAL.V_MMM_INPUT_WEEKLY', COUNT(*) FROM DIMENSIONAL.V_MMM_INPUT_WEEKLY;
"

# Check model output
echo ""
echo "--- Model Output ---"
snow sql -c $CONNECTION -q "
    USE ROLE ${DATABASE}_ROLE;
    USE DATABASE ${DATABASE};
    SELECT 'MODEL_RESULTS' as TBL, COUNT(*) as CNT FROM MMM.MODEL_RESULTS
    UNION ALL SELECT 'RESPONSE_CURVES', COUNT(*) FROM MMM.RESPONSE_CURVES
    UNION ALL SELECT 'MODEL_METADATA', COUNT(*) FROM MMM.MODEL_METADATA;
"

# Check model results details
echo ""
echo "--- Channel ROI Results ---"
snow sql -c $CONNECTION -q "
    USE ROLE ${DATABASE}_ROLE;
    USE DATABASE ${DATABASE};
    SELECT CHANNEL, 
           ROUND(ROI, 2) AS ROI, 
           ROUND(MARGINAL_ROI, 2) AS MROI,
           IS_SIGNIFICANT
    FROM MMM.MODEL_RESULTS
    ORDER BY ROI DESC;
"

# Check Streamlit URL
echo ""
echo "--- Streamlit URL ---"
snow streamlit get-url mmm_roi_app \
    -c $CONNECTION \
    --database $DATABASE \
    --schema MMM \
    --role ${DATABASE}_ROLE 2>/dev/null || echo "Streamlit not available"

echo ""
echo "=== Verification Complete ==="
```

Usage:
```bash
chmod +x verify_deployment.sh
./verify_deployment.sh demo
```

---

## 8. Troubleshooting

### 8.1 Common Issues

| Issue | Symptom | Diagnosis | Fix |
|-------|---------|-----------|-----|
| **Sandbox Error** | "unable to open lock file" / "Operation not permitted" | Running in sandboxed environment | Use `required_permissions: ["all"]` in Cursor AI |
| **Connection Fail** | `deploy.sh` fails at Step 1 | `snow connection test -c demo` | Check `~/.snowflake/config.toml` |
| **Compute Pool Full** | "unschedulable in full compute pool" | `SHOW SERVICES IN COMPUTE POOL ...` | `ALTER COMPUTE POOL ... STOP ALL;` |
| **Live Version Error** | "Live version not found" | Notebook not committed | `./deploy.sh --only-notebook` |
| **nevergrad Error** | "ModuleNotFoundError: nevergrad" | External access missing | Verify `PYPI_ACCESS_INTEGRATION` |
| **Column Mismatch** | "Insert value list does not match" | Schema mismatch | Check table DDL vs notebook output |
| **Data Missing** | Row counts = 0 | Check `data/synthetic/` | Generate data, then `--only-data` |
| **Streamlit 404** | URL returns "Not Found" | App not deployed | `./deploy.sh --only-streamlit` |
| **Unknown Channel** | "UNKNOWN_ALL_ALL" in results | Column mapping issue | Check `V_MMM_INPUT_WEEKLY` columns |

### 8.2 Debug Commands

```bash
# Test connection
snow connection test -c demo

# Check what's deployed
snow sql -c demo -q "SHOW DATABASES LIKE 'GLOBAL_B2B_MMM';"
snow sql -c demo -q "SHOW SCHEMAS IN DATABASE GLOBAL_B2B_MMM;"
snow sql -c demo -q "SHOW TABLES IN SCHEMA GLOBAL_B2B_MMM.ATOMIC;"

# Check compute pool
snow sql -c demo -q "DESCRIBE COMPUTE POOL GLOBAL_B2B_MMM_COMPUTE_POOL;"

# Stop stuck services
snow sql -c demo -q "
    USE ROLE ACCOUNTADMIN;
    ALTER COMPUTE POOL GLOBAL_B2B_MMM_COMPUTE_POOL STOP ALL;
"

# Check external access
snow sql -c demo -q "SHOW EXTERNAL ACCESS INTEGRATIONS LIKE 'PYPI%';"

# Check notebook
snow sql -c demo -q "SHOW NOTEBOOKS IN SCHEMA GLOBAL_B2B_MMM.MMM;"

# Check Streamlit
snow sql -c demo -q "SHOW STREAMLITS IN SCHEMA GLOBAL_B2B_MMM.MMM;"

# Check view column names
snow sql -c demo -q "DESCRIBE VIEW GLOBAL_B2B_MMM.DIMENSIONAL.V_MMM_INPUT_WEEKLY;"
```

### 8.3 Layered Debugging

Debug in layers when failures occur:

```
Layer 1: Prerequisites
   └── Is snow CLI installed and connected?
   └── Do local data files exist?

Layer 2: Infrastructure
   └── Did role, database, warehouse create?
   └── Is compute pool running (ACTIVE/IDLE)?
   └── Does external access integration exist?

Layer 3: Data
   └── Did files upload to stage?
   └── Did COPY INTO succeed?
   └── Are row counts correct?

Layer 4: Applications
   └── Did notebook deploy with live version?
   └── Did Streamlit deploy?

Layer 5: Execution
   └── Does notebook run without errors?
   └── Are MODEL_RESULTS populated?
   └── Is Streamlit accessible?
```

---

## 9. CI/CD Integration

### 9.1 GitHub Actions Example

```yaml
name: Snowflake Test Cycle

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install Snowflake CLI
        run: pip install snowflake-cli
      
      - name: Configure Connection
        run: |
          mkdir -p ~/.snowflake
          cat > ~/.snowflake/config.toml << EOF
          [connections.ci]
          account = "${{ secrets.SNOWFLAKE_ACCOUNT }}"
          user = "${{ secrets.SNOWFLAKE_USER }}"
          password = "${{ secrets.SNOWFLAKE_PASSWORD }}"
          EOF
      
      - name: Run Full Test Cycle
        run: |
          ./clean.sh -c ci --force
          ./deploy.sh -c ci
          ./run.sh -c ci main
        
      - name: Verify Results
        run: |
          COUNT=$(snow sql -c ci -q "
            SELECT COUNT(*) FROM GLOBAL_B2B_MMM.MMM.MODEL_RESULTS
          " -o tsv | tail -1)
          if [ "$COUNT" -ge 10 ]; then
            echo "✓ Test passed: $COUNT channels modeled"
          else
            echo "✗ Test failed: Only $COUNT channels"
            exit 1
          fi
```

### 9.2 CI Test Script

```bash
#!/bin/bash
###############################################################################
# ci_test.sh - Automated test for CI/CD
###############################################################################

set -e
CONNECTION="${SNOWFLAKE_CONNECTION:-ci}"
MIN_CHANNELS=10
TIMEOUT=600

echo "=== CI Test Cycle ==="

# Full cycle
./clean.sh -c $CONNECTION --force
./deploy.sh -c $CONNECTION
timeout $TIMEOUT ./run.sh -c $CONNECTION main

# Verify
COUNT=$(snow sql -c $CONNECTION -q "
    SELECT COUNT(*) FROM GLOBAL_B2B_MMM.MMM.MODEL_RESULTS
" -o tsv 2>/dev/null | tail -1)

if [ "$COUNT" -ge "$MIN_CHANNELS" ]; then
    echo "✓ PASSED: $COUNT channels modeled"
    exit 0
else
    echo "✗ FAILED: Only $COUNT channels (expected >= $MIN_CHANNELS)"
    exit 1
fi
```

---

## 10. Known Limitations

1. **No Sandbox Execution**: The Snowflake CLI requires write access to system cache directories (`~/Library/Caches/pyapp/locks/` on macOS). Scripts will fail in sandboxed environments like Cursor AI's default sandbox mode. Use full permissions when running via AI assistants.

2. **Environment Prefix**: `PYPI_ACCESS_INTEGRATION` references hardcoded database path. Prefixed deployments require manual external access setup.

3. **Dimensional Tables**: SCD2 dimensional tables (GEOGRAPHY, PRODUCT_CATEGORY, etc.) are not populated. Views use flat staging tables.

4. **Revenue Attribution**: `ATTRIBUTED_REVENUE` may show 0 if join keys between spend and revenue don't align.

5. **Segment Data**: `SEGMENT_NAME` is NULL in current data. Model uses "ALL" for product dimension.

6. **Compute Pool Capacity**: Single-node pool (`MAX_NODES=1`) may require stopping previous jobs before notebook execution.

---

## Quick Reference

### Full Cycle Commands

| Purpose | Command |
|---------|---------|
| Clean slate full cycle | `./clean.sh --force && ./deploy.sh && ./run.sh main` |
| Check status | `./run.sh status` |
| Get Streamlit URL | `./run.sh streamlit` |
| Verify deployment | `./verify_deployment.sh demo` |

### Component Deployments

| Flag | Deploys |
|------|---------|
| `--only-sql` | Account + Schema setup |
| `--only-data` | Upload + Load data |
| `--only-notebook` | Notebook deployment |
| `--only-streamlit` | Streamlit app |
| `--skip-notebook` | Full deploy minus notebook |

### Key Verification Queries

```sql
-- Quick health check
SELECT 
    (SELECT COUNT(*) FROM MMM.MODEL_RESULTS) as CHANNELS,
    (SELECT COUNT(*) FROM MMM.RESPONSE_CURVES) as CURVE_POINTS,
    (SELECT COUNT(*) FROM DIMENSIONAL.V_MMM_INPUT_WEEKLY) as INPUT_ROWS;

-- Channel ROI summary
SELECT CHANNEL, ROUND(ROI, 2) AS ROI, IS_SIGNIFICANT
FROM MMM.MODEL_RESULTS
ORDER BY ROI DESC;
```

---

**See Also:**
- [DRD.md](../DRD.md) - Design Reference Document
- [SNOWFLAKE_DEPLOYMENT_SCRIPT_GUIDELINES.md](../.cursor/SNOWFLAKE_DEPLOYMENT_SCRIPT_GUIDELINES.md)
- [SNOWFLAKE_DEMO_FULL_TEST_CYCLE.md](../.cursor/SNOWFLAKE_DEMO_FULL_TEST_CYCLE.md)
- [SNOWFLAKE_NOTEBOOK_GUIDELINES.md](../.cursor/SNOWFLAKE_NOTEBOOK_GUIDELINES.md)

