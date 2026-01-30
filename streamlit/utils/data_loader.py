from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import logging
import time
from typing import Dict

logger = logging.getLogger("snowflake.connector")
logger.setLevel(logging.INFO)


# =============================================================================
# Database Configuration
# =============================================================================
DATABASE = "GLOBAL_B2B_MMM"


# =============================================================================
# Centralized Query Definitions - Single source of truth for all SQL queries
# All queries use fully qualified names (DATABASE.SCHEMA.TABLE) for portability
# =============================================================================
QUERIES = {
    # Weekly input data for MMM - Note: View is in DIMENSIONAL schema
    "WEEKLY": f"SELECT * FROM {DATABASE}.DIMENSIONAL.V_MMM_INPUT_WEEKLY ORDER BY WEEK_START",
    
    # Response curves from model training
    "CURVES": f"SELECT * FROM {DATABASE}.MMM.RESPONSE_CURVES",
    
    # Model results - get all results (MODEL_RESULTS has no CREATED_AT column)
    "RESULTS": f"SELECT * FROM {DATABASE}.MMM.MODEL_RESULTS",
    
    # ROI summary by channel
    "ROI": f"SELECT * FROM {DATABASE}.MMM.V_ROI_BY_CHANNEL",
    
    # Model metadata - latest run
    "METADATA": f"SELECT * FROM {DATABASE}.MMM.MODEL_METADATA ORDER BY MODEL_RUN_DATE DESC LIMIT 1",
    
    # ROI by channel and region (from view)
    "ROI_REGION": f"SELECT * FROM {DATABASE}.MMM.V_ROI_BY_CHANNEL_REGION",
    
    # Regional aggregates from model results
    # Extracts region from CHANNEL name (e.g., 'Facebook_NA_ALL' -> 'NA')
    "RESULTS_BY_REGION": f"""
        SELECT 
            SPLIT_PART(CHANNEL, '_', -2) as REGION,
            AVG(ROI) as AVG_ROI,
            SUM(CURRENT_SPEND) as TOTAL_SPEND,
            AVG(MARGINAL_ROI) as AVG_MARGINAL_ROI,
            COUNT(*) as CHANNEL_COUNT,
            ARRAY_AGG(CHANNEL) as CHANNELS
        FROM {DATABASE}.MMM.MODEL_RESULTS
        GROUP BY SPLIT_PART(CHANNEL, '_', -2)
        ORDER BY AVG_ROI DESC
    """,
}


def run_queries_parallel(
    session,
    queries: Dict[str, str],
    max_workers: int = 4,
    return_empty_on_error: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Execute multiple independent SQL queries in parallel.
    
    Args:
        session: Snowflake Snowpark session
        queries: Dict mapping names to SQL strings
        max_workers: Max concurrent queries (4 recommended for Snowflake)
        return_empty_on_error: Return empty DataFrame on failure vs raise
    
    Returns:
        Dict mapping query names to result DataFrames
    """
    if not queries:
        logger.info("[DATA_LOADER] No queries provided")
        return {}
    
    logger.info(f"[DATA_LOADER] Starting parallel execution of {len(queries)} queries")
    for name, query in queries.items():
        logger.info(f"[DATA_LOADER] Query '{name}': {query[:100]}...")
    
    start_time = time.time()
    results: Dict[str, pd.DataFrame] = {}
    
    def execute_query(name: str, query: str) -> tuple:
        try:
            logger.info(f"[DATA_LOADER] Executing query '{name}'")
            df = session.sql(query).to_pandas()
            logger.info(f"[DATA_LOADER] Query '{name}' returned {len(df)} rows, columns: {list(df.columns)}")
            return name, df
        except Exception as e:
            logger.error(f"[DATA_LOADER] Query '{name}' FAILED: {type(e).__name__}: {e}")
            if return_empty_on_error:
                return name, pd.DataFrame()
            raise
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {
            executor.submit(execute_query, name, query): name
            for name, query in queries.items()
        }
        
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                query_name, result_df = future.result()
                results[query_name] = result_df
                logger.info(f"[DATA_LOADER] Stored result for '{query_name}': {len(result_df)} rows")
            except Exception as e:
                logger.error(f"[DATA_LOADER] Future for '{name}' failed: {e}")
                if return_empty_on_error:
                    results[name] = pd.DataFrame()
                else:
                    raise
    
    elapsed = time.time() - start_time
    logger.info(f"[DATA_LOADER] Parallel execution complete: {len(queries)} queries in {elapsed:.2f}s")
    for name, df in results.items():
        logger.info(f"[DATA_LOADER] Final result '{name}': {len(df)} rows, empty={df.empty}")
    
    return results

