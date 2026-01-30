#!/bin/bash
###############################################################################
# deploy.sh - Deploy Global B2B MMM Demo to Snowflake
#
# Operations:
#   1. Check prerequisites (snow CLI, connection)
#   2. Run Account-Level SQL (Role, DB, WH, Compute Pool)
#   3. Run Schema-Level SQL (Tables, Stages, Views)
#   4. Upload Synthetic Data to Stage
#   5. Load Data into Tables
#   6. Deploy Notebook
#   7. Deploy Streamlit App
#
# Usage:
#   ./deploy.sh                  # Default usage
#   ./deploy.sh -c demo          # With connection
#   ./deploy.sh --prefix DEV     # With environment prefix
###############################################################################

set -e
set -o pipefail

# Configuration
CONNECTION_NAME=""  # Empty = use snowcli default connection
SKIP_NOTEBOOK=false
ENV_PREFIX=""
ONLY_COMPONENT=""

PROJECT_PREFIX="GLOBAL_B2B_MMM"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy the Global B2B MMM Demo project.

Options:
  -c, --connection NAME    Snowflake CLI connection name (default: snowcli default)
  -p, --prefix PREFIX      Environment prefix for resources (e.g., DEV, PROD)
  --skip-notebook          Skip notebook deployment
  --only-streamlit         Deploy only the Streamlit app
  --only-notebook          Deploy only the Notebook
  --only-data              Upload and load data only
  --only-sql               Run SQL setup only
  -h, --help               Show this help message

Examples:
  $0                       # Full deployment (uses snowcli default connection)
  $0 -c aws3               # Use 'aws3' connection
  $0 --prefix DEV          # Deploy with DEV_ prefix
EOF
    exit 0
}

error_exit() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) usage ;;
        -c|--connection) CONNECTION_NAME="$2"; shift 2 ;;
        -p|--prefix) ENV_PREFIX="$2"; shift 2 ;;
        --skip-notebook) SKIP_NOTEBOOK=true; shift ;;
        --only-streamlit) ONLY_COMPONENT="streamlit"; shift ;;
        --only-notebook) ONLY_COMPONENT="notebook"; shift ;;
        --only-data) ONLY_COMPONENT="data"; shift ;;
        --only-sql) ONLY_COMPONENT="sql"; shift ;;
        *) error_exit "Unknown option: $1" ;;
    esac
done

# Build connection argument (empty if using default)
if [ -n "$CONNECTION_NAME" ]; then
    SNOW_CONN="-c $CONNECTION_NAME"
else
    SNOW_CONN=""
fi

# Compute resource names
if [ -n "$ENV_PREFIX" ]; then
    FULL_PREFIX="${ENV_PREFIX}_${PROJECT_PREFIX}"
else
    FULL_PREFIX="${PROJECT_PREFIX}"
fi

DATABASE="${FULL_PREFIX}"
SCHEMA="ATOMIC" # Default schema for setup, though we use RAW/MMM too
ROLE="${FULL_PREFIX}_ROLE"
WAREHOUSE="${FULL_PREFIX}_WH"
COMPUTE_POOL="${FULL_PREFIX}_COMPUTE_POOL"
# Network/External Access not used in this demo, but placeholders for pattern
NETWORK_RULE="${FULL_PREFIX}_EGRESS_RULE" 
EXTERNAL_ACCESS="${FULL_PREFIX}_EXTERNAL_ACCESS"

# Display configuration banner
echo "=================================================="
echo "Global B2B MMM - Deployment"
echo "=================================================="
echo "Configuration:"
echo "  Connection: ${CONNECTION_NAME:-<default>}"
echo "  Prefix: ${ENV_PREFIX:-<none>}"
echo "  Database: $DATABASE"
echo "  Role: $ROLE"
echo "  Warehouse: $WAREHOUSE"
echo "  Compute Pool: $COMPUTE_POOL"
echo ""

should_run_step() {
    local step_name="$1"
    if [ -z "$ONLY_COMPONENT" ]; then
        return 0
    fi
    case "$ONLY_COMPONENT" in
        sql)
            [[ "$step_name" == "account_sql" || "$step_name" == "schema_sql" ]]
            ;;
        data)
            [[ "$step_name" == "upload_data" || "$step_name" == "load_data" ]]
            ;;
        notebook)
            [[ "$step_name" == "notebook" ]]
            ;;
        streamlit)
            [[ "$step_name" == "streamlit" ]]
            ;;
        *)
            return 1
            ;;
    esac
}

# Step 1: Check prerequisites
echo "Step 1: Checking prerequisites..."
if ! command -v snow &> /dev/null; then
    error_exit "Snowflake CLI (snow) not found."
fi

echo "Testing Snowflake connection..."
if ! snow sql $SNOW_CONN -q "SELECT 1" &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Failed to connect to Snowflake"
    snow connection test $SNOW_CONN 2>&1 || true
    exit 1
fi
echo -e "${GREEN}[OK]${NC} Connection verified"

# Step 2: Account-Level SQL
if should_run_step "account_sql"; then
    echo "Step 2: Running account-level SQL setup..."
    {
        echo "-- Set session variables"
        echo "SET FULL_PREFIX = '${FULL_PREFIX}';"
        echo "SET PROJECT_ROLE = '${ROLE}';"
        echo "SET PROJECT_WH = '${WAREHOUSE}';"
        echo "SET PROJECT_COMPUTE_POOL = '${COMPUTE_POOL}';"
        echo "SET PROJECT_SCHEMA = '${SCHEMA}';"
        echo ""
        cat sql/01_account_setup.sql
    } | snow sql $SNOW_CONN -i
    echo -e "${GREEN}[OK]${NC} Account setup complete"
fi

# Step 3: Schema-Level SQL
if should_run_step "schema_sql"; then
    echo "Step 3: Running schema-level SQL setup..."
    {
        echo "USE ROLE ${ROLE};"
        echo "USE DATABASE ${DATABASE};"
        echo "USE WAREHOUSE ${WAREHOUSE};"
        echo ""
        cat sql/02_schema_setup.sql
        echo ""
        cat sql/04_cortex_setup.sql
    } | snow sql $SNOW_CONN -i
    echo -e "${GREEN}[OK]${NC} Schema setup complete"
    
    # Grant CREATE MODEL for Model Registry (requires ACCOUNTADMIN)
    echo "Step 3b: Granting Model Registry privileges..."
    snow sql $SNOW_CONN -q "
        USE ROLE ACCOUNTADMIN;
        GRANT CREATE MODEL ON SCHEMA ${DATABASE}.MMM TO ROLE ${ROLE};
        GRANT CREATE MODEL ON SCHEMA ${DATABASE}.ATOMIC TO ROLE ${ROLE};
    " 2>/dev/null || echo "  Note: CREATE MODEL grant may require manual ACCOUNTADMIN setup"
    echo -e "${GREEN}[OK]${NC} Model Registry privileges configured"
fi

# Step 4: Upload Data
if should_run_step "upload_data"; then
    echo "Step 4: Uploading synthetic data..."
    
    # Check if data exists, if not generate it
    if [ ! -f "data/synthetic/sprinklr_spend.csv" ]; then
        echo "Generating synthetic data..."
        python utils/generate_synthetic_data.py
    fi
    
    snow sql $SNOW_CONN -q "
        USE ROLE ${ROLE};
        USE DATABASE ${DATABASE};
        USE SCHEMA ATOMIC;
        PUT file://data/synthetic/*.csv @DATA_STAGE AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
        PUT file://data/synthetic/campaign_briefs/*.pdf @DATA_STAGE/campaign_briefs/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
    "
    echo -e "${GREEN}[OK]${NC} Data uploaded"
fi

# Step 5: Load Data
if should_run_step "load_data"; then
    echo "Step 5: Loading data into tables..."
    {
        echo "USE ROLE ${ROLE};"
        echo "USE DATABASE ${DATABASE};"
        echo "USE WAREHOUSE ${WAREHOUSE};"
        echo ""
        cat sql/03_load_data.sql
    } | snow sql $SNOW_CONN -i
    echo -e "${GREEN}[OK]${NC} Data loaded"
fi

# Step 6: Deploy Notebooks
if should_run_step "notebook" && [ "$SKIP_NOTEBOOK" = false ]; then
    echo "Step 6: Deploying Notebook..."
    
    # Upload notebook file
    snow sql $SNOW_CONN -q "
        USE ROLE ${ROLE};
        USE DATABASE ${DATABASE};
        USE SCHEMA ATOMIC;
        PUT file://notebooks/01_mmm_training.ipynb @MODELS_STAGE/notebooks/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
    "
    
    # Create notebook: MMM Training
    snow sql $SNOW_CONN -q "
        USE ROLE ${ROLE};
        USE DATABASE ${DATABASE};
        USE SCHEMA MMM;
        
        CREATE OR REPLACE NOTEBOOK MMM_TRAINING_NOTEBOOK
            FROM '@ATOMIC.MODELS_STAGE/notebooks/'
            MAIN_FILE = '01_mmm_training.ipynb'
            RUNTIME_NAME = 'SYSTEM\$BASIC_RUNTIME'
            COMPUTE_POOL = '${COMPUTE_POOL}'
            QUERY_WAREHOUSE = '${WAREHOUSE}'
            EXTERNAL_ACCESS_INTEGRATIONS = (PYPI_ACCESS_INTEGRATION)
            COMMENT = 'MMM Training Notebook - trains model and registers to Model Registry';
            
        ALTER NOTEBOOK MMM_TRAINING_NOTEBOOK ADD LIVE VERSION FROM LAST;
    "
    echo -e "${GREEN}[OK]${NC} MMM Training Notebook deployed"
fi

# Step 7: Deploy Streamlit
if should_run_step "streamlit"; then
    echo "Step 7: Deploying Streamlit app on Container Runtime..."
    
    # First, upload the app files to stage
    echo "  Uploading Streamlit files to stage..."
    snow sql $SNOW_CONN -q "
        USE ROLE ${ROLE};
        USE DATABASE ${DATABASE};
        USE SCHEMA MMM;
        CREATE STAGE IF NOT EXISTS STREAMLIT_STAGE;
    "
    
    # Upload all Streamlit files
    snow sql $SNOW_CONN -q "
        USE ROLE ${ROLE};
        USE DATABASE ${DATABASE};
        USE SCHEMA MMM;
        PUT file://streamlit/mmm_roi_app.py @STREAMLIT_STAGE/mmm_roi_app/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
        PUT file://streamlit/requirements.txt @STREAMLIT_STAGE/mmm_roi_app/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
    "
    
    # Upload pages and utils directories
    for file in streamlit/pages/*.py; do
        snow sql $SNOW_CONN -q "
            USE ROLE ${ROLE};
            USE DATABASE ${DATABASE};
            USE SCHEMA MMM;
            PUT file://$file @STREAMLIT_STAGE/mmm_roi_app/pages/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
        "
    done
    
    for file in streamlit/utils/*.py; do
        snow sql $SNOW_CONN -q "
            USE ROLE ${ROLE};
            USE DATABASE ${DATABASE};
            USE SCHEMA MMM;
            PUT file://$file @STREAMLIT_STAGE/mmm_roi_app/utils/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
        "
    done
    
    # Create Streamlit app with Container Runtime
    echo "  Creating Streamlit app with Container Runtime..."
    snow sql $SNOW_CONN -q "
        USE ROLE ${ROLE};
        USE DATABASE ${DATABASE};
        USE SCHEMA MMM;
        
        DROP STREAMLIT IF EXISTS MMM_ROI_APP;
        
        CREATE STREAMLIT MMM_ROI_APP
            FROM '@STREAMLIT_STAGE/mmm_roi_app/'
            MAIN_FILE = 'mmm_roi_app.py'
            RUNTIME_NAME = 'SYSTEM\$ST_CONTAINER_RUNTIME_PY3_11'
            COMPUTE_POOL = ${COMPUTE_POOL}
            QUERY_WAREHOUSE = ${WAREHOUSE}
            EXTERNAL_ACCESS_INTEGRATIONS = (PYPI_ACCESS_INTEGRATION)
            COMMENT = 'Global B2B MMM ROI Engine - Marketing Mix Model Analysis';
    "
    
    echo -e "${GREEN}[OK]${NC} Streamlit app deployed on Container Runtime"
fi

echo ""
echo "=================================================="
echo -e "${GREEN}Deployment Complete!${NC}"
echo "=================================================="
echo ""
echo "--- Deployment Summary ---"
if [ -z "$ONLY_COMPONENT" ]; then
    echo "Roles: ${ROLE}"
    echo "Warehouses: ${WAREHOUSE}"
    echo "Databases: ${DATABASE}"
    echo "Compute Pools: ${COMPUTE_POOL}"
    echo "Schemas: 4 (RAW, ATOMIC, MMM, DIMENSIONAL)"
    echo "Tables: 17"
    echo "Views: 11"
    echo "Stages: 3"
    echo "Streamlit Apps: 1"
    echo "Notebooks: 1"
else
    case "$ONLY_COMPONENT" in
        sql)
            echo "Roles: ${ROLE}"
            echo "Warehouses: ${WAREHOUSE}"
            echo "Databases: ${DATABASE}"
            echo "Compute Pools: ${COMPUTE_POOL}"
            echo "Schemas: 4"
            echo "Tables: 17"
            echo "Views: 11"
            ;;
        data)
            echo "Data files uploaded and loaded"
            ;;
        notebook)
            echo "Notebooks: 1"
            ;;
        streamlit)
            echo "Streamlit Apps: 1"
            ;;
    esac
fi
echo ""
echo "Next Steps:"
echo "  1. Run the Notebook to train the model:"
echo "     ./run.sh main"
echo ""
echo "  2. Open the Streamlit App:"
echo "     ./run.sh streamlit"

