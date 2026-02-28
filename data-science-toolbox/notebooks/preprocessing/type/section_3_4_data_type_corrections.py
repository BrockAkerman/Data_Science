"""
Section 3.4: Data Type Corrections

Systematic data type standardization and validation.
Runs AFTER missing value handling and BEFORE outlier detection.

Handles:
- Categorical encoding (convert strings to category dtype)
- Numeric consistency (int vs float standardization)
- Type validation and error handling
- Comprehensive audit logging
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Any, List

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_TYPE_CONFIG = {
    'numeric_columns': {
        'customer_age': 'int64',
        'customer_income': 'int64',
        'employment_duration': 'float64',
        'loan_amnt': 'float64',
        'loan_int_rate': 'float64',
        'term_years': 'int64',
        'cred_hist_length': 'int64',
    },
    'categorical_columns': {
        'home_ownership': 'category',      # Categorical nominal
        'loan_intent': 'category',         # Categorical nominal
        'loan_grade': 'category',          # Categorical ordinal (A > B > C > D...)
        'historical_default': 'category',  # Categorical binary/nominal
        'Current_loan_status': 'category', # Categorical nominal
    },
    'columns_to_drop': [
        'customer_id',  # Already removed in duplicates section
    ],
    'validate_ranges': {
        'customer_age': {'min': 18, 'max': 100},           # Domain bounds (post-cleaning)
        'customer_income': {'min': 0, 'max': None},        # Must be non-negative
        'loan_amnt': {'min': 0, 'max': 900000},            # Domain bounds (post-cleaning)
        'loan_int_rate': {'min': 0, 'max': 100},           # Percentage (0-100)
        'term_years': {'min': 1, 'max': 50},               # Reasonable loan term
        'cred_hist_length': {'min': 0, 'max': 80},         # Credit history (0-80 years)
    }
}


# ==============================================================================
# ANALYSIS FUNCTION
# ==============================================================================

def analyze_data_types(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze current data types before correction.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    config : Dict
        Data type configuration
    
    Returns:
    --------
    Dict with analysis results
    """
    analysis = {
        'current_dtypes': df.dtypes.to_dict(),
        'numeric_columns': [],
        'categorical_columns': [],
        'object_columns': [],
        'columns_to_drop': [],
        'warnings': []
    }
    
    logger.info("\n" + "="*70)
    logger.info("DATA TYPE ANALYSIS (BEFORE CORRECTION)")
    logger.info("="*70)
    
    logger.info(f"\nCurrent dtypes:")
    for col, dtype in df.dtypes.items():
        logger.info(f"  {col:30s}: {str(dtype):15s}")
    
    # Identify column types
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            analysis['numeric_columns'].append(col)
        elif pd.api.types.is_object_dtype(df[col]):
            analysis['object_columns'].append(col)
    
    logger.info(f"\nColumn classification:")
    logger.info(f"  Numeric columns: {len(analysis['numeric_columns'])}")
    logger.info(f"  Object/String columns: {len(analysis['object_columns'])}")
    
    # Check for columns that should be dropped
    if 'columns_to_drop' in config:
        for col in config['columns_to_drop']:
            if col in df.columns:
                analysis['columns_to_drop'].append(col)
                logger.info(f"  Will drop: {col}")
    
    logger.info(f"\n" + "="*70)
    return analysis


# ==============================================================================
# CORRECTION FUNCTION
# ==============================================================================

def correct_data_types(
    df: pd.DataFrame,
    config: Dict[str, Any] = None,
    analysis: Dict[str, Any] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Correct and standardize data types across all columns.
    
    Handles:
    - Converting numeric columns to specified dtypes (int64, float64)
    - Converting categorical columns to category dtype
    - Dropping unnecessary columns
    - Validating value ranges
    - Comprehensive audit logging
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe (post-missing-values, pre-outliers)
    config : Dict, optional
        Data type configuration
    analysis : Dict, optional
        Pre-computed analysis
    
    Returns:
    --------
    df_clean : pd.DataFrame
        Corrected dataframe with proper dtypes
    audit : Dict
        Comprehensive audit trail
    """
    if config is None:
        config = DATA_TYPE_CONFIG
    
    if analysis is None:
        analysis = analyze_data_types(df, config)
    
    audit = {
        'status': 'STARTED',
        'rows_input': len(df),
        'rows_output': len(df),
        'columns_input': len(df.columns),
        'columns_output': len(df.columns),
        'columns_dropped': [],
        'dtype_conversions': [],
        'validation_errors': [],
        'warnings': [],
        'details': {}
    }
    
    df_clean = df.copy()
    
    logger.info("\n" + "="*70)
    logger.info("DATA TYPE CORRECTION EXECUTION")
    logger.info("="*70)
    
    # =========================================================================
    # STEP 1: DROP UNNECESSARY COLUMNS
    # =========================================================================
    logger.info("\nStep 1: Dropping unnecessary columns")
    logger.info(f"{'─'*70}")
    
    if 'columns_to_drop' in config:
        for col in config['columns_to_drop']:
            if col in df_clean.columns:
                df_clean = df_clean.drop(columns=[col])
                audit['columns_dropped'].append(col)
                logger.info(f"  ✓ Dropped: {col}")
    
    if not audit['columns_dropped']:
        logger.info(f"  No columns to drop")
    
    # =========================================================================
    # STEP 2: CONVERT NUMERIC COLUMNS
    # =========================================================================
    logger.info(f"\nStep 2: Converting numeric columns")
    logger.info(f"{'─'*70}")
    
    if 'numeric_columns' in config:
        for col, target_dtype in config['numeric_columns'].items():
            if col not in df_clean.columns:
                continue
            
            current_dtype = df_clean[col].dtype
            
            if current_dtype == target_dtype:
                logger.info(f"  ✓ {col:30s}: Already {target_dtype}")
                continue
            
            try:
                df_clean[col] = df_clean[col].astype(target_dtype)
                audit['dtype_conversions'].append({
                    'column': col,
                    'from': str(current_dtype),
                    'to': target_dtype,
                    'success': True
                })
                logger.info(f"  ✓ {col:30s}: {str(current_dtype):15s} → {target_dtype}")
            
            except (ValueError, TypeError) as e:
                error_msg = f"Failed to convert {col} to {target_dtype}: {str(e)}"
                audit['validation_errors'].append(error_msg)
                logger.error(f"  ✗ {col:30s}: ERROR - {error_msg}")
    
    # =========================================================================
    # STEP 3: CONVERT CATEGORICAL COLUMNS
    # =========================================================================
    logger.info(f"\nStep 3: Converting categorical columns")
    logger.info(f"{'─'*70}")
    
    if 'categorical_columns' in config:
        for col, target_dtype in config['categorical_columns'].items():
            if col not in df_clean.columns:
                continue
            
            current_dtype = df_clean[col].dtype
            
            try:
                df_clean[col] = df_clean[col].astype(target_dtype)
                n_categories = df_clean[col].nunique()
                audit['dtype_conversions'].append({
                    'column': col,
                    'from': str(current_dtype),
                    'to': target_dtype,
                    'n_categories': n_categories,
                    'success': True
                })
                logger.info(f"  ✓ {col:30s}: {str(current_dtype):15s} → {target_dtype} ({n_categories} categories)")
            
            except (ValueError, TypeError) as e:
                error_msg = f"Failed to convert {col} to {target_dtype}: {str(e)}"
                audit['validation_errors'].append(error_msg)
                logger.error(f"  ✗ {col:30s}: ERROR - {error_msg}")
    
    # =========================================================================
    # STEP 4: VALIDATE VALUE RANGES
    # =========================================================================
    logger.info(f"\nStep 4: Validating value ranges")
    logger.info(f"{'─'*70}")
    
    if 'validate_ranges' in config:
        for col, bounds in config['validate_ranges'].items():
            if col not in df_clean.columns:
                continue
            
            min_val = bounds.get('min')
            max_val = bounds.get('max')
            
            violations = 0
            violation_details = {'column': col, 'violations': {}}
            
            if min_val is not None:
                below_min = (df_clean[col] < min_val).sum()
                violations += below_min
                if below_min > 0:
                    violation_details['violations']['below_min'] = below_min
                    logger.warning(f"  ⚠ {col:30s}: {below_min} values below {min_val}")
            
            if max_val is not None:
                above_max = (df_clean[col] > max_val).sum()
                violations += above_max
                if above_max > 0:
                    violation_details['violations']['above_max'] = above_max
                    logger.warning(f"  ⚠ {col:30s}: {above_max} values above {max_val}")
            
            if violations == 0:
                logger.info(f"  ✓ {col:30s}: All values within [{min_val}, {max_val}]")
            else:
                audit['warnings'].append(violation_details)
    
    # =========================================================================
    # STEP 5: FINAL SUMMARY
    # =========================================================================
    audit['rows_output'] = len(df_clean)
    audit['columns_output'] = len(df_clean.columns)
    
    logger.info(f"\n" + "="*70)
    logger.info("DATA TYPE CORRECTION SUMMARY")
    logger.info("="*70)
    logger.info(f"Status: SUCCESS")
    logger.info(f"Columns dropped: {len(audit['columns_dropped'])}")
    if audit['columns_dropped']:
        logger.info(f"  → {', '.join(audit['columns_dropped'])}")
    logger.info(f"Dtype conversions: {len(audit['dtype_conversions'])}")
    logger.info(f"Validation warnings: {len(audit['warnings'])}")
    logger.info(f"Validation errors: {len(audit['validation_errors'])}")
    logger.info(f"Rows: {audit['rows_input']} → {audit['rows_output']} (dropped: {audit['rows_input'] - audit['rows_output']})")
    logger.info(f"Columns: {audit['columns_input']} → {audit['columns_output']} (dropped: {len(audit['columns_dropped'])})")
    logger.info(f"="*70)
    
    if audit['validation_errors']:
        audit['status'] = 'FAILED'
        logger.error(f"\nValidation failed with {len(audit['validation_errors'])} errors")
    else:
        audit['status'] = 'SUCCESS'
    
    return df_clean, audit


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    # After missing value handling, before outlier detection:
    
    # Step 1: Analyze
    analysis = analyze_data_types(data, DATA_TYPE_CONFIG)
    
    # Step 2: Correct
    data_typed, audit = correct_data_types(data, DATA_TYPE_CONFIG, analysis)
    
    # Step 3: Use cleaned data in next pipeline step
    outlier_analysis = analyze_outliers_all(data_typed, OUTLIER_CONFIG)
