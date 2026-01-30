#!/bin/bash
###############################################################################
# clean.sh - Remove all Global B2B MMM resources from Snowflake
###############################################################################

set -e
set -o pipefail

CONNECTION_NAME=""
ENV_PREFIX=""
DRY_RUN=false
FORCE=false

PROJECT_PREFIX="GLOBAL_B2B_MMM"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Remove all Global B2B MMM project resources from Snowflake.

Options:
  -c, --connection NAME    Snowflake CLI connection name (default: snowcli default)
  -p, --prefix PREFIX      Environment prefix for resources (e.g., DEV, PROD)
  --dry-run                Show what would be dropped without executing
  --force, -y              Skip confirmation prompt
  -h, --help               Show this help message

Examples:
  $0                       # Full cleanup (uses snowcli default connection)
  $0 -c aws3               # Use 'aws3' connection
  $0 --prefix DEV          # Cleanup DEV_ prefixed resources
  $0 --dry-run             # Preview cleanup without executing
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) usage ;;
        -c|--connection) CONNECTION_NAME="$2"; shift 2 ;;
        -p|--prefix) ENV_PREFIX="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --force|-y) FORCE=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -n "$CONNECTION_NAME" ]; then
    SNOW_CONN="-c $CONNECTION_NAME"
else
    SNOW_CONN=""
fi

if [ -n "$ENV_PREFIX" ]; then
    FULL_PREFIX="${ENV_PREFIX}_${PROJECT_PREFIX}"
else
    FULL_PREFIX="${PROJECT_PREFIX}"
fi

DATABASE="${FULL_PREFIX}"
ROLE="${FULL_PREFIX}_ROLE"
WAREHOUSE="${FULL_PREFIX}_WH"
COMPUTE_POOL="${FULL_PREFIX}_COMPUTE_POOL"

echo "=================================================="
echo "Global B2B MMM - Cleanup"
echo "=================================================="
echo "Configuration:"
echo "  Connection: ${CONNECTION_NAME:-<default>}"
echo "  Prefix: ${ENV_PREFIX:-<none>}"
echo "  Database: $DATABASE"
echo "  Role: $ROLE"
echo "  Warehouse: $WAREHOUSE"
echo "  Compute Pool: $COMPUTE_POOL"
if [ "$DRY_RUN" = true ]; then
    echo -e "  ${YELLOW}Mode: DRY RUN (no changes will be made)${NC}"
fi
echo ""

if [ "$DRY_RUN" = false ] && [ "$FORCE" = false ]; then
    echo -e "${YELLOW}WARNING: This will permanently delete all project resources!${NC}"
    read -p "Are you sure you want to continue? (yes/no): " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo "Cleanup cancelled."
        exit 0
    fi
fi

echo "Verifying Snowflake connection..."
if ! snow sql $SNOW_CONN -q "SELECT 1" &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Failed to connect to Snowflake"
    snow connection test $SNOW_CONN 2>&1 || true
    exit 1
fi
echo -e "${GREEN}[OK]${NC} Connection verified"
echo ""

run_sql() {
    local sql="$1"
    local desc="$2"
    if [ "$DRY_RUN" = true ]; then
        echo -e "  ${YELLOW}[DRY RUN]${NC} Would execute: $desc"
    else
        echo "  $desc..."
        snow sql $SNOW_CONN -q "$sql" 2>/dev/null && echo -e "  ${GREEN}[OK]${NC}" || echo -e "  ${YELLOW}[WARN]${NC} May not exist"
    fi
}

echo "Dropping Streamlit App..."
run_sql "USE ROLE ${ROLE}; DROP STREAMLIT IF EXISTS ${DATABASE}.MMM.MMM_ROI_APP;" "DROP STREAMLIT MMM_ROI_APP"

echo "Dropping Notebook..."
run_sql "USE ROLE ${ROLE}; DROP NOTEBOOK IF EXISTS ${DATABASE}.MMM.MMM_TRAINING_NOTEBOOK;" "DROP NOTEBOOK MMM_TRAINING_NOTEBOOK"

echo "Dropping External Access Integration..."
run_sql "USE ROLE ACCOUNTADMIN; DROP INTEGRATION IF EXISTS PYPI_ACCESS_INTEGRATION;" "DROP INTEGRATION PYPI_ACCESS_INTEGRATION"

echo "Dropping Database (cascades schemas, tables, views, stages)..."
run_sql "USE ROLE ACCOUNTADMIN; DROP DATABASE IF EXISTS ${DATABASE} CASCADE;" "DROP DATABASE ${DATABASE}"

echo "Dropping Compute Pool..."
run_sql "USE ROLE ACCOUNTADMIN; ALTER COMPUTE POOL IF EXISTS ${COMPUTE_POOL} STOP ALL; DROP COMPUTE POOL IF EXISTS ${COMPUTE_POOL};" "DROP COMPUTE POOL ${COMPUTE_POOL}"

echo "Dropping Warehouse..."
run_sql "USE ROLE ACCOUNTADMIN; DROP WAREHOUSE IF EXISTS ${WAREHOUSE};" "DROP WAREHOUSE ${WAREHOUSE}"

echo "Dropping Role..."
run_sql "USE ROLE ACCOUNTADMIN; DROP ROLE IF EXISTS ${ROLE};" "DROP ROLE ${ROLE}"

echo ""
echo "=================================================="
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}Dry Run Complete - No changes were made${NC}"
else
    echo -e "${GREEN}Cleanup Complete!${NC}"
fi
echo "=================================================="
echo ""
echo "--- Cleanup Summary ---"
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
echo "Integrations: 1 (PYPI_ACCESS_INTEGRATION)"
