#!/bin/bash
###############################################################################
# run.sh - Runtime operations for Global B2B MMM Demo
#
# Commands:
#   main       - Execute the MMM Training Notebook
#   status     - Check resource status
#   streamlit  - Get Streamlit app URL
###############################################################################

set -e
set -o pipefail

CONNECTION_NAME=""  # Empty = use snowcli default connection
COMMAND=""
ENV_PREFIX=""

PROJECT_PREFIX="GLOBAL_B2B_MMM"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Commands:
  main       Execute the MMM Training Notebook
  status     Check resource status
  streamlit  Get Streamlit App URL

Options:
  -c, --connection NAME    Snowflake CLI connection name (default: snowcli default)
  -p, --prefix PREFIX      Environment prefix
  -h, --help               Show this help
EOF
    exit 0
}

error_exit() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) usage ;;
        -c|--connection) CONNECTION_NAME="$2"; shift 2 ;;
        -p|--prefix) ENV_PREFIX="$2"; shift 2 ;;
        main|status|streamlit) COMMAND="$1"; shift ;;
        *) error_exit "Unknown option: $1" ;;
    esac
done

[ -z "$COMMAND" ] && usage

# Build connection argument (empty if using default)
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
SCHEMA="MMM" # Mart schema where apps/notebooks live
ROLE="${FULL_PREFIX}_ROLE"
WAREHOUSE="${FULL_PREFIX}_WH"
COMPUTE_POOL="${FULL_PREFIX}_COMPUTE_POOL"

cmd_main() {
    echo "=================================================="
    echo "Executing MMM Training Notebook"
    echo "=================================================="
    
    echo "Triggering notebook execution..."
    if ! snow sql $SNOW_CONN -q "
        USE ROLE ${ROLE};
        USE DATABASE ${DATABASE};
        USE SCHEMA MMM;
        EXECUTE NOTEBOOK MMM_TRAINING_NOTEBOOK();
    "; then
        error_exit "Notebook execution failed"
    fi
    
    echo -e "${GREEN}[OK]${NC} Notebook executed successfully."
}



cmd_status() {
    echo "=================================================="
    echo "Global B2B MMM - Status"
    echo "=================================================="
    
    echo "Compute Pool:"
    snow sql $SNOW_CONN -q "SHOW COMPUTE POOLS LIKE '${COMPUTE_POOL}';" \
        2>/dev/null || echo "  Not found"
        
    echo "Warehouse:"
    snow sql $SNOW_CONN -q "USE ROLE ${ROLE}; SHOW WAREHOUSES LIKE '${WAREHOUSE}';" \
        2>/dev/null || echo "  Not found"
        
    echo "Row Counts:"
    snow sql $SNOW_CONN -q "
        USE ROLE ${ROLE};
        USE DATABASE ${DATABASE};
        SELECT 'Media Spend' as TBL, COUNT(*) as CNT FROM ATOMIC.MEDIA_SPEND_DAILY
        UNION ALL
        SELECT 'Opportunities', COUNT(*) FROM ATOMIC.OPPORTUNITY;
    " 2>/dev/null || echo "  Error querying tables"
}

cmd_streamlit() {
    echo "=================================================="
    echo "Global B2B MMM - Streamlit URL"
    echo "=================================================="
    
    URL=$(snow streamlit get-url mmm_roi_app \
        $SNOW_CONN \
        --database $DATABASE \
        --schema MMM \
        --role $ROLE 2>/dev/null) || true
        
    if [ -n "$URL" ]; then
        echo "Dashboard URL:"
        echo "  $URL"
    else
        echo "Could not retrieve URL. Check if deployed."
    fi
}

case $COMMAND in
    main) cmd_main ;;
    status) cmd_status ;;
    streamlit) cmd_streamlit ;;
    *) error_exit "Unknown command" ;;
esac

