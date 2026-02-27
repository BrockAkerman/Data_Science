"""
Data Cleaning Pipeline: Duplicate Handling Module

This module provides production-ready duplicate detection and removal
with comprehensive audit logging.

Author: [Your Name]
Date: 2024
"""

import pandas as pd
import logging
from typing import Dict, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEFAULT_DUPLICATE_CONFIG = {
    'check_exact_rows': True,
    'check_customer_level': True,
    'customer_id_column': 'customer_id',
    'keep': 'first',
    'fail_if_duplicates_remain': True,
    'subset': None,
}


# ==============================================================================
# DUPLICATE HANDLING FUNCTION
# ==============================================================================

def handle_duplicates(
    df: pd.DataFrame, 
    config: Dict[str, Any] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Detect and remove duplicate records at both exact-row and customer-level.
    
    This function performs comprehensive duplicate detection and removal with
    detailed audit logging suitable for production data pipelines.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe to check for duplicates
    
    config : Dict, optional
        Configuration dict with the following keys:
        
        - 'check_exact_rows' (bool): Check for exact row duplicates
          Default: True
        
        - 'check_customer_level' (bool): Check for customer-level duplicates
          Default: True
        
        - 'customer_id_column' (str): Name of the customer ID column
          Default: 'customer_id'
        
        - 'keep' (str): Which duplicate to keep ('first', 'last', False)
          Default: 'first'
        
        - 'fail_if_duplicates_remain' (bool): Raise error if duplicates remain
          Default: True
        
        - 'subset' (list or None): Columns to consider for duplication check
          Default: None (all columns)
    
    Returns:
    --------
    df_clean : pd.DataFrame
        Dataframe with duplicates removed
    
    audit : Dict
        Comprehensive audit trail with keys:
        
        - 'status': 'SUCCESS' or 'FAILED'
        - 'rows_input': Number of input rows
        - 'rows_output': Number of output rows
        - 'rows_removed': Number of rows removed
        - 'checks_performed': List of checks executed
        - 'duplicate_details': Dict with duplicate detection results
        - 'errors': List of any errors encountered
    
    Raises:
    -------
    ValueError
        If duplicates remain after removal and fail_if_duplicates_remain=True
    
    Examples:
    ---------
    >>> import pandas as pd
    >>> from duplicate_handling import handle_duplicates
    >>> 
    >>> # Load your data
    >>> df = pd.read_csv('loan_data.csv')
    >>> 
    >>> # Define config
    >>> config = {
    ...     'check_exact_rows': True,
    ...     'check_customer_level': True,
    ...     'customer_id_column': 'customer_id',
    ...     'keep': 'first',
    ...     'fail_if_duplicates_remain': True,
    ... }
    >>> 
    >>> # Run pipeline
    >>> df_clean, audit = handle_duplicates(df, config)
    >>> 
    >>> # Check results
    >>> print(f"Rows removed: {audit['rows_removed']}")
    >>> print(f"Status: {audit['status']}")
    """
    
    if config is None:
        config = DEFAULT_DUPLICATE_CONFIG
    
    # Initialize audit trail
    audit = {
        'status': 'STARTED',
        'rows_input': len(df),
        'rows_removed': 0,
        'checks_performed': [],
        'duplicate_details': {},
        'errors': []
    }
    
    df_clean = df.copy()
    
    # =========================================================================
    # CHECK 1: EXACT ROW DUPLICATES
    # =========================================================================
    if config['check_exact_rows']:
        n_exact_dupes = df_clean.duplicated(subset=config['subset'], keep=False).sum()
        audit['checks_performed'].append('exact_row_check')
        
        if n_exact_dupes > 0:
            # Get duplicate rows for inspection
            dup_mask = df_clean.duplicated(subset=config['subset'], keep=False)
            dup_rows = df_clean[dup_mask].sort_values(
                by=list(df_clean.columns[:5])
            ).reset_index(drop=True)
            
            audit['duplicate_details']['exact_duplicates'] = {
                'count': int(n_exact_dupes),
                'duplicate_pairs': int(len(df_clean[dup_mask]) // 2),
                'sample_indices': dup_rows.index.tolist()[:6]
            }
            
            logger.info(f"✓ Exact row check: Found {n_exact_dupes} duplicate rows")
            logger.info(f"  Sample of duplicates (first 6 rows):")
            logger.info(dup_rows.head(6))
        else:
            audit['duplicate_details']['exact_duplicates'] = {'count': 0}
            logger.info(f"✓ Exact row check: No exact duplicates found")
    
    # =========================================================================
    # CHECK 2: CUSTOMER-LEVEL DUPLICATES
    # =========================================================================
    if config['check_customer_level'] and config['customer_id_column'] in df_clean.columns:
        customer_col = config['customer_id_column']
        n_unique_customers = df_clean[customer_col].nunique()
        n_total_rows = len(df_clean)
        n_customer_dupes = n_total_rows - n_unique_customers
        
        audit['checks_performed'].append('customer_level_check')
        
        if n_customer_dupes > 0:
            # Identify which customers appear multiple times
            dup_customer_mask = df_clean[customer_col].duplicated(keep=False)
            dup_customers = df_clean[dup_customer_mask].sort_values(customer_col)
            
            # Get summary of duplicates per customer
            dupes_per_customer = df_clean[customer_col].value_counts()
            dupes_per_customer = dupes_per_customer[dupes_per_customer > 1]
            
            audit['duplicate_details']['customer_level'] = {
                'duplicate_customers': int(len(dupes_per_customer)),
                'duplicate_rows': int(n_customer_dupes),
                'distribution': dupes_per_customer.to_dict(),
            }
            
            logger.info(f"✓ Customer-level check: {len(dupes_per_customer)} customers appear multiple times")
            logger.info(f"  Total rows involved: {n_customer_dupes}")
            logger.info(f"  Distribution: {dict(dupes_per_customer)}")
            logger.info(f"  Sample of duplicated customers:")
            logger.info(dup_customers.head(8))
        else:
            audit['duplicate_details']['customer_level'] = {
                'duplicate_customers': 0,
                'duplicate_rows': 0
            }
            logger.info(f"✓ Customer-level check: No customer-level duplicates found")
    
    # =========================================================================
    # REMOVAL: DROP DUPLICATES
    # =========================================================================
    rows_before = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=config['subset'], keep=config['keep'])
    rows_removed = rows_before - len(df_clean)
    
    audit['rows_removed'] = int(rows_removed)
    audit['rows_output'] = len(df_clean)
    
    if rows_removed > 0:
        logger.info(f"\n✓ Duplicate removal: {rows_removed} rows removed")
        logger.info(f"  Input rows: {rows_before} → Output rows: {len(df_clean)}")
    else:
        logger.info(f"\n✓ No duplicates to remove")
    
    # =========================================================================
    # VALIDATION: Confirm no duplicates remain
    # =========================================================================
    remaining_exact_dupes = df_clean.duplicated(subset=config['subset']).sum()
    
    if remaining_exact_dupes > 0:
        error_msg = f"Duplicate removal failed: {remaining_exact_dupes} duplicates remain"
        audit['errors'].append(error_msg)
        audit['status'] = 'FAILED'
        logger.error(f"✗ VALIDATION FAILED: {error_msg}")
        
        if config['fail_if_duplicates_remain']:
            raise ValueError(error_msg)
    else:
        audit['status'] = 'SUCCESS'
        logger.info(f"\n✓ VALIDATION PASSED: No duplicates remain")
    
    # Print summary
    logger.info(f"\n" + "="*70)
    logger.info(f"DUPLICATE HANDLING SUMMARY")
    logger.info(f"="*70)
    logger.info(f"Status: {audit['status']}")
    logger.info(f"Rows removed: {audit['rows_removed']}")
    logger.info(f"Input shape: ({audit['rows_input']}, {df.shape[1]})")
    logger.info(f"Output shape: ({audit['rows_output']}, {df_clean.shape[1]})")
    logger.info(f"="*70)
    
    return df_clean, audit


# ==============================================================================
# PIPELINE CLASS (Template for full pipeline)
# ==============================================================================

class DataCleaningPipeline:
    """
    Orchestrates the full data cleaning pipeline.
    
    This class coordinates multiple cleaning steps (duplicates, missing values,
    outliers) and maintains a comprehensive audit trail for each step.
    
    Attributes:
    -----------
    config : Dict
        Pipeline configuration
    
    audit_trail : Dict
        Audit logs for each cleaning step
    
    Methods:
    --------
    execute(df) -> (DataFrame, Dict)
        Run all cleaning steps in sequence
    
    get_summary() -> str
        Print formatted summary of cleaning results
    
    Example:
    --------
    >>> config = {
    ...     'duplicate_handling': {...},
    ...     'missing_value_handling': {...},
    ...     'outlier_handling': {...}
    ... }
    >>> 
    >>> pipeline = DataCleaningPipeline(config)
    >>> df_clean, audit = pipeline.execute(df_raw)
    >>> print(pipeline.get_summary())
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline with configuration.
        
        Parameters:
        -----------
        config : Dict
            Configuration dict with keys for each cleaning step:
            - 'duplicate_handling'
            - 'missing_value_handling' [future]
            - 'outlier_handling' [future]
        """
        self.config = config
        self.audit_trail = {}
    
    def execute(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run all cleaning steps in sequence.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw input dataframe
        
        Returns:
        --------
        df_clean : pd.DataFrame
            Cleaned dataframe
        
        audit_trail : Dict
            Comprehensive audit log of all cleaning steps
        """
        df_current = df.copy()
        
        # Step 1: Handle duplicates
        logger.info("\n" + "#"*70)
        logger.info("# STEP 1: DUPLICATE HANDLING")
        logger.info("#"*70)
        df_current, audit = handle_duplicates(df_current, self.config['duplicate_handling'])
        self.audit_trail['duplicates'] = audit
        
        # Step 2: Handle missing values [FUTURE]
        # df_current, audit = handle_missing_values(df_current, ...)
        # self.audit_trail['missing_values'] = audit
        
        # Step 3: Handle outliers [FUTURE]
        # df_current, audit = handle_outliers(df_current, ...)
        # self.audit_trail['outliers'] = audit
        
        return df_current, self.audit_trail
    
    def get_summary(self) -> str:
        """
        Return formatted summary of pipeline execution.
        
        Returns:
        --------
        str
            Formatted summary of cleaning results
        """
        summary = "\n" + "="*70 + "\n"
        summary += "PIPELINE EXECUTION SUMMARY\n"
        summary += "="*70 + "\n"
        
        for step_name, step_audit in self.audit_trail.items():
            summary += f"\n{step_name.upper()}:\n"
            summary += f"  Status: {step_audit['status']}\n"
            summary += f"  Rows removed: {step_audit['rows_removed']}\n"
            summary += f"  Checks: {', '.join(step_audit['checks_performed'])}\n"
        
        summary += "\n" + "="*70
        return summary
