#!/bin/bash

###############################################################################
# create_user.sh - Create a Snowflake user with access to a demo project
#
# This script automatically infers project configuration from deploy.sh in the
# current project directory. All inferred values can be overridden via CLI.
#
# Usage:
#   ./create_user.sh --user USERNAME [OPTIONS]
#
# Required:
#   --user, -u NAME           Username to create
#
# Optional (auto-inferred from project if not specified):
#   --connection, -c NAME     Snowflake CLI connection name
#   --database, -d NAME       Project database name
#   --schema, -s NAME         Project schema name
#   --role, -r NAME           Role name
#   --warehouse, -w NAME      Warehouse name
#   --compute-pool NAME       Compute pool name (for notebook access)
#   --prefix PREFIX           Environment prefix (e.g., DEV, PROD)
#
# Other Options:
#   --password, -p PASS       Initial password (if not set, user must use SSO)
#   --email EMAIL             User's email address
#   --first-name NAME         User's first name
#   --last-name NAME          User's last name
#   --comment TEXT            Comment for the user
#   --no-change-password      Do NOT force password change on first login
#   --dry-run                 Show SQL without executing
#   --show-config             Show inferred configuration and exit
#   -h, --help                Show this help message
#
# Examples:
#   # Minimal - just specify username (everything else inferred from project)
#   ./create_user.sh -u demo_user -p TempPass123!
#
#   # Override connection
#   ./create_user.sh -u demo_user -c prod -p TempPass123!
#
#   # Use environment prefix (matches deploy.sh --prefix option)
#   ./create_user.sh -u demo_user --prefix DEV -p TempPass123!
#
#   # Show what would be inferred
#   ./create_user.sh --show-config
#
#   # Dry run to see SQL
#   ./create_user.sh -u test_user --dry-run
###############################################################################

set -e
set -o pipefail

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

###############################################################################
# Auto-detect project configuration from deploy.sh
###############################################################################

# Defaults (will be overridden by deploy.sh if present)
DEFAULT_CONNECTION_NAME="demo"
DEFAULT_PROJECT_PREFIX=""
DEFAULT_ENV_PREFIX=""

# Try to read project settings from deploy.sh
if [ -f "$SCRIPT_DIR/deploy.sh" ]; then
    # Extract PROJECT_PREFIX from deploy.sh
    DETECTED_PROJECT_PREFIX=$(grep -E '^PROJECT_PREFIX=' "$SCRIPT_DIR/deploy.sh" 2>/dev/null | head -1 | sed 's/PROJECT_PREFIX=["'"'"']*\([^"'"'"']*\)["'"'"']*/\1/')
    if [ -n "$DETECTED_PROJECT_PREFIX" ]; then
        DEFAULT_PROJECT_PREFIX="$DETECTED_PROJECT_PREFIX"
    fi
    
    # Extract default CONNECTION_NAME from deploy.sh
    DETECTED_CONNECTION=$(grep -E '^CONNECTION_NAME=' "$SCRIPT_DIR/deploy.sh" 2>/dev/null | head -1 | sed 's/CONNECTION_NAME=["'"'"']*\([^"'"'"']*\)["'"'"']*/\1/')
    if [ -n "$DETECTED_CONNECTION" ]; then
        DEFAULT_CONNECTION_NAME="$DETECTED_CONNECTION"
    fi
fi

# Initialize with defaults (can be overridden by CLI)
CONNECTION_NAME=""
USER_NAME=""
SNOWFLAKE_DATABASE=""
SNOWFLAKE_SCHEMA=""
SNOWFLAKE_ROLE=""
SNOWFLAKE_WAREHOUSE=""
COMPUTE_POOL_NAME=""
ENV_PREFIX=""
PASSWORD=""
EMAIL=""
FIRST_NAME=""
LAST_NAME=""
COMMENT=""
MUST_CHANGE_PASSWORD="TRUE"
DRY_RUN=false
SHOW_CONFIG=false

# Function to display usage
usage() {
    cat << EOF
Usage: $0 --user USERNAME [OPTIONS]

Create a Snowflake user with access to a demo project.

This script auto-detects project configuration from deploy.sh in the current
directory. Detected values can be overridden via command line options.

Required:
  -u, --user NAME           Username to create

Auto-Inferred (override with CLI if needed):
  -c, --connection NAME     Snowflake CLI connection name
  -d, --database NAME       Project database name
  -s, --schema NAME         Project schema name
  -r, --role NAME           Role name
  -w, --warehouse NAME      Warehouse name
  --compute-pool NAME       Compute pool name (for notebook access)
  --prefix PREFIX           Environment prefix (e.g., DEV, PROD)

Other Options:
  -p, --password PASS       Initial password (if not set, user must use SSO)
  --email EMAIL             User's email address
  --first-name NAME         User's first name
  --last-name NAME          User's last name
  --comment TEXT            Comment for the user
  --no-change-password      Do NOT force password change on first login
  --dry-run                 Show SQL without executing
  --show-config             Show inferred configuration and exit
  -h, --help                Show this help message

EOF

    # Show what's auto-detected
    echo "Auto-Detected Configuration (from deploy.sh):"
    if [ -n "$DEFAULT_PROJECT_PREFIX" ]; then
        echo "  Project Prefix:    $DEFAULT_PROJECT_PREFIX"
        echo "  Connection:        $DEFAULT_CONNECTION_NAME"
        echo "  Database:          $DEFAULT_PROJECT_PREFIX"
        echo "  Schema:            $DEFAULT_PROJECT_PREFIX"
        echo "  Role:              ${DEFAULT_PROJECT_PREFIX}_ROLE"
        echo "  Warehouse:         ${DEFAULT_PROJECT_PREFIX}_WH"
        echo "  Compute Pool:      ${DEFAULT_PROJECT_PREFIX}_COMPUTE_POOL"
    else
        echo "  (No deploy.sh found - specify all parameters manually)"
    fi
    echo ""

    cat << EOF
Examples:
  $0 -u demo_user -p TempPass123!           # Minimal - infer everything
  $0 -u demo_user -c prod -p TempPass123!   # Override connection
  $0 -u demo_user --prefix DEV              # Use DEV environment prefix
  $0 --show-config                          # Show detected configuration
  $0 -u test_user --dry-run                 # Preview SQL
EOF
    exit 0
}

# Error exit function
error_exit() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        -u|--user)
            USER_NAME="$2"
            shift 2
            ;;
        -c|--connection)
            CONNECTION_NAME="$2"
            shift 2
            ;;
        -d|--database)
            SNOWFLAKE_DATABASE="$2"
            shift 2
            ;;
        -s|--schema)
            SNOWFLAKE_SCHEMA="$2"
            shift 2
            ;;
        -r|--role)
            SNOWFLAKE_ROLE="$2"
            shift 2
            ;;
        -w|--warehouse)
            SNOWFLAKE_WAREHOUSE="$2"
            shift 2
            ;;
        --compute-pool)
            COMPUTE_POOL_NAME="$2"
            shift 2
            ;;
        --prefix)
            ENV_PREFIX="$2"
            shift 2
            ;;
        -p|--password)
            PASSWORD="$2"
            shift 2
            ;;
        --email)
            EMAIL="$2"
            shift 2
            ;;
        --first-name)
            FIRST_NAME="$2"
            shift 2
            ;;
        --last-name)
            LAST_NAME="$2"
            shift 2
            ;;
        --comment)
            COMMENT="$2"
            shift 2
            ;;
        --no-change-password)
            MUST_CHANGE_PASSWORD="FALSE"
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --show-config)
            SHOW_CONFIG=true
            shift
            ;;
        *)
            error_exit "Unknown option: $1\nUse --help for usage information"
            ;;
    esac
done

###############################################################################
# Apply Defaults and Build Configuration
###############################################################################

# Apply default connection if not specified
if [ -z "$CONNECTION_NAME" ]; then
    CONNECTION_NAME="$DEFAULT_CONNECTION_NAME"
fi

# Compute the full prefix (may include environment prefix)
if [ -z "$DEFAULT_PROJECT_PREFIX" ]; then
    # No deploy.sh found - require explicit parameters
    if [ -z "$SNOWFLAKE_DATABASE" ] || [ -z "$SNOWFLAKE_SCHEMA" ]; then
        error_exit "No deploy.sh found. Please specify --database and --schema explicitly.\nUse --help for usage information"
    fi
    FULL_PREFIX="$SNOWFLAKE_DATABASE"
else
    # Use project prefix, optionally with environment prefix
    if [ -n "$ENV_PREFIX" ]; then
        FULL_PREFIX="${ENV_PREFIX}_${DEFAULT_PROJECT_PREFIX}"
    else
        FULL_PREFIX="${DEFAULT_PROJECT_PREFIX}"
    fi
fi

# Apply inferred values for any unspecified parameters
if [ -z "$SNOWFLAKE_DATABASE" ]; then
    SNOWFLAKE_DATABASE="$FULL_PREFIX"
fi

if [ -z "$SNOWFLAKE_SCHEMA" ]; then
    # Default to MMM schema for this project (Analytics Mart)
    SNOWFLAKE_SCHEMA="MMM"
fi

if [ -z "$SNOWFLAKE_ROLE" ]; then
    SNOWFLAKE_ROLE="${FULL_PREFIX}_ROLE"
fi

if [ -z "$SNOWFLAKE_WAREHOUSE" ]; then
    SNOWFLAKE_WAREHOUSE="${FULL_PREFIX}_WH"
fi

if [ -z "$COMPUTE_POOL_NAME" ]; then
    # Only set compute pool if we have a project prefix
    if [ -n "$DEFAULT_PROJECT_PREFIX" ]; then
        COMPUTE_POOL_NAME="${FULL_PREFIX}_COMPUTE_POOL"
    fi
fi

if [ -z "$COMMENT" ]; then
    COMMENT="${SNOWFLAKE_DATABASE} Demo User"
fi

###############################################################################
# Show Config Mode
###############################################################################
if [ "$SHOW_CONFIG" = true ]; then
    echo "=================================================="
    echo "Inferred Project Configuration"
    echo "=================================================="
    echo ""
    if [ -f "$SCRIPT_DIR/deploy.sh" ]; then
        echo -e "${GREEN}[OK]${NC} deploy.sh found in project directory"
    else
        echo -e "${YELLOW}[WARN]${NC} No deploy.sh found - using defaults"
    fi
    echo ""
    echo "Configuration that will be used:"
    echo "  Connection:     $CONNECTION_NAME"
    echo "  Database:       $SNOWFLAKE_DATABASE"
    echo "  Schema:         $SNOWFLAKE_SCHEMA"
    echo "  Role:           $SNOWFLAKE_ROLE"
    echo "  Warehouse:      $SNOWFLAKE_WAREHOUSE"
    if [ -n "$COMPUTE_POOL_NAME" ]; then
        echo "  Compute Pool:   $COMPUTE_POOL_NAME"
    fi
    if [ -n "$ENV_PREFIX" ]; then
        echo "  Env Prefix:     $ENV_PREFIX"
    fi
    echo ""
    echo "Override any value with CLI options. Use --help for details."
    echo ""
    exit 0
fi

###############################################################################
# Validate Required Parameters
###############################################################################
if [ -z "$USER_NAME" ]; then
    error_exit "Missing required parameter: --user\nUse --help for usage information"
fi

# Validate username format (alphanumeric and underscore only)
if ! [[ "$USER_NAME" =~ ^[A-Za-z][A-Za-z0-9_]*$ ]]; then
    error_exit "Invalid username format. Must start with a letter and contain only letters, numbers, and underscores."
fi

# Convert username to uppercase (Snowflake convention)
USER_NAME_UPPER=$(echo "$USER_NAME" | tr '[:lower:]' '[:upper:]')

echo "=================================================="
echo "Snowflake Demo - Create User"
echo "=================================================="
echo ""
echo "Configuration (auto-inferred from project):"
echo "  User:       $USER_NAME_UPPER"
echo "  Database:   $SNOWFLAKE_DATABASE"
echo "  Schema:     $SNOWFLAKE_SCHEMA"
echo "  Role:       $SNOWFLAKE_ROLE"
echo "  Warehouse:  $SNOWFLAKE_WAREHOUSE"
if [ -n "$COMPUTE_POOL_NAME" ]; then
    echo "  Compute Pool: $COMPUTE_POOL_NAME"
fi
if [ -n "$ENV_PREFIX" ]; then
    echo "  Env Prefix: $ENV_PREFIX"
fi
echo ""
echo -e "${BLUE}[TIP]${NC} Use --show-config to see all inferred values"
echo ""

###############################################################################
# Step 1: Check Prerequisites
###############################################################################
echo "Step 1: Checking prerequisites..."
echo "------------------------------------------------"

# Check if snow CLI is installed
if ! command -v snow &> /dev/null; then
    error_exit "snow CLI not found. Install with: pip install snowflake-cli"
fi
echo -e "${GREEN}[OK]${NC} Snowflake CLI found"

# Test actual connection with a simple query
echo "Testing Snowflake connection..."
CONNECTION_TEST=$(snow sql -c "$CONNECTION_NAME" -q "SELECT CURRENT_USER()" 2>&1)
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR]${NC} Failed to connect to Snowflake"
    echo ""
    echo "Connection test output:"
    echo "$CONNECTION_TEST"
    echo ""
    echo "Possible causes:"
    echo "  - JWT private key passphrase not set"
    echo "  - Invalid credentials"
    echo "  - Network connectivity issues"
    echo ""
    echo "For JWT authentication, ensure you've set the passphrase:"
    echo "  export SNOWFLAKE_PRIVATE_KEY_PASSPHRASE='your_passphrase'"
    echo ""
    exit 1
fi
echo -e "${GREEN}[OK]${NC} Connection '$CONNECTION_NAME' verified"

# Verify that the project is deployed (database exists)
DB_CHECK=$(snow sql -c "$CONNECTION_NAME" -q "SHOW DATABASES LIKE '$SNOWFLAKE_DATABASE'" 2>&1)
if ! echo "$DB_CHECK" | grep -q "$SNOWFLAKE_DATABASE"; then
    error_exit "Database '$SNOWFLAKE_DATABASE' not found. Deploy the project first with ./deploy.sh"
fi
echo -e "${GREEN}[OK]${NC} Database '$SNOWFLAKE_DATABASE' exists"

# Check if user already exists
USER_EXISTS=false
USER_CHECK=$(snow sql -c "$CONNECTION_NAME" -q "SHOW USERS LIKE '$USER_NAME_UPPER'" 2>&1)
if echo "$USER_CHECK" | grep -qi "$USER_NAME_UPPER"; then
    USER_EXISTS=true
    echo -e "${YELLOW}[INFO]${NC} User '$USER_NAME_UPPER' already exists - will grant project access"
fi

echo ""

###############################################################################
# Step 2: Build SQL Commands
###############################################################################
echo "Step 2: Building SQL commands..."
echo "------------------------------------------------"

# Build optional user properties (only used for new users)
USER_OPTIONS=""
if [ -n "$PASSWORD" ]; then
    USER_OPTIONS="${USER_OPTIONS} PASSWORD = '${PASSWORD}'"
fi
if [ -n "$EMAIL" ]; then
    USER_OPTIONS="${USER_OPTIONS} EMAIL = '${EMAIL}'"
fi
if [ -n "$FIRST_NAME" ]; then
    USER_OPTIONS="${USER_OPTIONS} FIRST_NAME = '${FIRST_NAME}'"
fi
if [ -n "$LAST_NAME" ]; then
    USER_OPTIONS="${USER_OPTIONS} LAST_NAME = '${LAST_NAME}'"
fi

# Build compute pool grants if specified
COMPUTE_POOL_GRANTS=""
if [ -n "$COMPUTE_POOL_NAME" ]; then
    COMPUTE_POOL_GRANTS="
-- ============================================================
-- GRANT COMPUTE POOL ACCESS (for notebook execution)
-- ============================================================

GRANT USAGE ON COMPUTE POOL ${COMPUTE_POOL_NAME} TO ROLE ${SNOWFLAKE_ROLE};
GRANT MONITOR ON COMPUTE POOL ${COMPUTE_POOL_NAME} TO ROLE ${SNOWFLAKE_ROLE};
"
fi

# Build user creation SQL (only if user doesn't exist)
USER_CREATE_SQL=""
if [ "$USER_EXISTS" = false ]; then
    USER_CREATE_SQL="
-- ============================================================
-- 1. CREATE USER
-- ============================================================

CREATE USER IF NOT EXISTS ${USER_NAME_UPPER}
    ${USER_OPTIONS}
    MUST_CHANGE_PASSWORD = ${MUST_CHANGE_PASSWORD}
    DEFAULT_WAREHOUSE = '${SNOWFLAKE_WAREHOUSE}'
    DEFAULT_NAMESPACE = '${SNOWFLAKE_DATABASE}.${SNOWFLAKE_SCHEMA}'
    DEFAULT_ROLE = '${SNOWFLAKE_ROLE}'
    COMMENT = '${COMMENT}';
"
fi

# Build the SQL header based on whether user exists
if [ "$USER_EXISTS" = true ]; then
    SQL_HEADER="GRANT ACCESS TO USER: ${USER_NAME_UPPER}"
else
    SQL_HEADER="CREATE USER: ${USER_NAME_UPPER}"
fi

# Build the SQL script
SQL_SCRIPT=$(cat << EOF
-- ============================================================
-- ${SQL_HEADER}
-- For: ${SNOWFLAKE_DATABASE} Demo
-- ============================================================

USE ROLE ACCOUNTADMIN;
${USER_CREATE_SQL}
-- ============================================================
-- GRANT PROJECT ROLE TO USER
-- ============================================================

GRANT ROLE ${SNOWFLAKE_ROLE} TO USER ${USER_NAME_UPPER};

-- ============================================================
-- GRANT WAREHOUSE USAGE
-- ============================================================

GRANT USAGE ON WAREHOUSE ${SNOWFLAKE_WAREHOUSE} TO USER ${USER_NAME_UPPER};

-- ============================================================
-- 4. GRANT DATABASE ACCESS
-- ============================================================

GRANT USAGE ON DATABASE ${SNOWFLAKE_DATABASE} TO ROLE ${SNOWFLAKE_ROLE};
GRANT USAGE ON SCHEMA ${SNOWFLAKE_DATABASE}.${SNOWFLAKE_SCHEMA} TO ROLE ${SNOWFLAKE_ROLE};

-- ============================================================
-- 5. GRANT TABLE ACCESS
-- ============================================================

GRANT SELECT ON ALL TABLES IN SCHEMA ${SNOWFLAKE_DATABASE}.${SNOWFLAKE_SCHEMA} TO ROLE ${SNOWFLAKE_ROLE};
GRANT SELECT ON FUTURE TABLES IN SCHEMA ${SNOWFLAKE_DATABASE}.${SNOWFLAKE_SCHEMA} TO ROLE ${SNOWFLAKE_ROLE};

-- ============================================================
-- 6. GRANT VIEW ACCESS
-- ============================================================

GRANT SELECT ON ALL VIEWS IN SCHEMA ${SNOWFLAKE_DATABASE}.${SNOWFLAKE_SCHEMA} TO ROLE ${SNOWFLAKE_ROLE};
GRANT SELECT ON FUTURE VIEWS IN SCHEMA ${SNOWFLAKE_DATABASE}.${SNOWFLAKE_SCHEMA} TO ROLE ${SNOWFLAKE_ROLE};

-- ============================================================
-- 7. GRANT STAGE ACCESS
-- ============================================================

GRANT READ ON ALL STAGES IN SCHEMA ${SNOWFLAKE_DATABASE}.${SNOWFLAKE_SCHEMA} TO ROLE ${SNOWFLAKE_ROLE};
GRANT READ ON FUTURE STAGES IN SCHEMA ${SNOWFLAKE_DATABASE}.${SNOWFLAKE_SCHEMA} TO ROLE ${SNOWFLAKE_ROLE};

-- ============================================================
-- 8. GRANT FUNCTION ACCESS
-- ============================================================

GRANT USAGE ON ALL FUNCTIONS IN SCHEMA ${SNOWFLAKE_DATABASE}.${SNOWFLAKE_SCHEMA} TO ROLE ${SNOWFLAKE_ROLE};
GRANT USAGE ON FUTURE FUNCTIONS IN SCHEMA ${SNOWFLAKE_DATABASE}.${SNOWFLAKE_SCHEMA} TO ROLE ${SNOWFLAKE_ROLE};

-- ============================================================
-- 9. GRANT STREAMLIT ACCESS
-- ============================================================

GRANT USAGE ON ALL STREAMLITS IN SCHEMA ${SNOWFLAKE_DATABASE}.${SNOWFLAKE_SCHEMA} TO ROLE ${SNOWFLAKE_ROLE};
GRANT USAGE ON FUTURE STREAMLITS IN SCHEMA ${SNOWFLAKE_DATABASE}.${SNOWFLAKE_SCHEMA} TO ROLE ${SNOWFLAKE_ROLE};
${COMPUTE_POOL_GRANTS}
-- ============================================================
-- 10. GRANT CORTEX LLM ACCESS
-- ============================================================

GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE ${SNOWFLAKE_ROLE};

-- ============================================================
-- 11. GRANT FILE FORMAT ACCESS
-- ============================================================

GRANT USAGE ON ALL FILE FORMATS IN SCHEMA ${SNOWFLAKE_DATABASE}.${SNOWFLAKE_SCHEMA} TO ROLE ${SNOWFLAKE_ROLE};

-- ============================================================
-- VERIFICATION
-- ============================================================

DESCRIBE USER ${USER_NAME_UPPER};
SHOW GRANTS TO USER ${USER_NAME_UPPER};

SELECT 'User ${USER_NAME_UPPER} created successfully!' AS status;
EOF
)

###############################################################################
# Step 3: Execute or Display SQL
###############################################################################
if [ "$DRY_RUN" = true ]; then
    echo ""
    echo -e "${YELLOW}[DRY RUN] The following SQL would be executed:${NC}"
    echo "=================================================="
    echo ""
    echo "$SQL_SCRIPT"
    echo ""
    echo "=================================================="
    echo -e "${YELLOW}[DRY RUN] No changes were made.${NC}"
    echo ""
    exit 0
fi

if [ "$USER_EXISTS" = true ]; then
    echo "Granting project access to existing user '${USER_NAME_UPPER}'..."
else
    echo "Creating user '${USER_NAME_UPPER}'..."
fi
echo ""

###############################################################################
# Step 3: Execute SQL
###############################################################################
echo "Step 3: Executing SQL..."
echo "------------------------------------------------"

# Execute the SQL
snow sql -c "$CONNECTION_NAME" -q "$SQL_SCRIPT"

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    if [ "$USER_EXISTS" = true ]; then
        echo -e "${GREEN}[OK] Project Access Granted Successfully!${NC}"
    else
        echo -e "${GREEN}[OK] User Created Successfully!${NC}"
    fi
    echo "=================================================="
    echo ""
    
    # Retrieve account information
    echo "Retrieving account information..."
    ACCOUNT_INFO=$(snow sql -c "$CONNECTION_NAME" -q "SELECT CURRENT_ACCOUNT() AS account, CURRENT_ORGANIZATION_NAME() AS org, CURRENT_REGION() AS region" --format json 2>/dev/null || echo "[]")
    
    # Parse account info
    ACCOUNT_NAME=$(echo "$ACCOUNT_INFO" | grep -o '"ACCOUNT"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed 's/.*: *"\([^"]*\)".*/\1/' || echo "")
    ORG_NAME=$(echo "$ACCOUNT_INFO" | grep -o '"ORG"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed 's/.*: *"\([^"]*\)".*/\1/' || echo "")
    
    # Build account URL
    if [ -n "$ORG_NAME" ] && [ -n "$ACCOUNT_NAME" ]; then
        ACCOUNT_URL="https://app.snowflake.com/${ORG_NAME}/${ACCOUNT_NAME}"
        ACCOUNT_IDENTIFIER="${ORG_NAME}-${ACCOUNT_NAME}"
    elif [ -n "$ACCOUNT_NAME" ]; then
        ACCOUNT_URL="https://${ACCOUNT_NAME}.snowflakecomputing.com"
        ACCOUNT_IDENTIFIER="${ACCOUNT_NAME}"
    else
        ACCOUNT_URL="[Could not retrieve - check connection]"
        ACCOUNT_IDENTIFIER="[Check with administrator]"
    fi
    
    echo ""
    echo "============================================================"
    echo "  USER ACCESS INFORMATION"
    echo "============================================================"
    echo ""
    echo "SNOWFLAKE LOGIN"
    echo "---------------"
    echo "  Web Login URL:      ${ACCOUNT_URL}"
    echo "  Account Identifier: ${ACCOUNT_IDENTIFIER}"
    echo "  Username:           ${USER_NAME_UPPER}"
    if [ "$USER_EXISTS" = true ]; then
        echo "  Password:           [Existing user - use current credentials]"
    elif [ -n "$PASSWORD" ]; then
        echo "  Temporary Password: ${PASSWORD}"
        if [ "$MUST_CHANGE_PASSWORD" = "TRUE" ]; then
            echo "  (Password change required on first login)"
        fi
    else
        echo "  Password:           [Contact administrator for SSO setup]"
    fi
    echo ""
    echo "PROJECT DETAILS"
    echo "---------------"
    echo "  Database:           ${SNOWFLAKE_DATABASE}"
    echo "  Schema:             ${SNOWFLAKE_DATABASE}.${SNOWFLAKE_SCHEMA}"
    echo "  Role:               ${SNOWFLAKE_ROLE}"
    echo "  Warehouse:          ${SNOWFLAKE_WAREHOUSE}"
    if [ -n "$COMPUTE_POOL_NAME" ]; then
        echo "  Compute Pool:       ${COMPUTE_POOL_NAME}"
    fi
    echo ""
    echo "QUICK START SQL"
    echo "---------------"
    echo "  USE ROLE ${SNOWFLAKE_ROLE};"
    echo "  USE DATABASE ${SNOWFLAKE_DATABASE};"
    echo "  USE SCHEMA ${SNOWFLAKE_SCHEMA};"
    echo "  USE WAREHOUSE ${SNOWFLAKE_WAREHOUSE};"
    echo ""
    echo "============================================================"
    echo ""
    echo "Admin Notes:"
    echo "  To remove this user later:"
    echo "    snow sql -c ${CONNECTION_NAME} -q \"DROP USER IF EXISTS ${USER_NAME_UPPER};\""
    echo ""
    echo "  To reset password:"
    echo "    snow sql -c ${CONNECTION_NAME} -q \"ALTER USER ${USER_NAME_UPPER} SET PASSWORD = 'NewPassword!';\""
    echo ""
else
    if [ "$USER_EXISTS" = true ]; then
        error_exit "Failed to grant project access. Check the error messages above."
    else
        error_exit "Failed to create user. Check the error messages above."
    fi
fi
