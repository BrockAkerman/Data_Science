"""
Section 3.5: Value Corrections

Standardize and clean categorical values, fix impossible values,
and handle remaining data quality issues after type conversion.

Runs AFTER Section 3.4 (Data Type Corrections) and BEFORE Section 3.6 (Validation).
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

VALUE_CORRECTION_CONFIG = {
    # Categorical value standardization (case, spacing, etc.)
    'categorical_standardization': {
        'home_ownership': {
            'mapping': {
                # Define any case/spacing standardization needed
                # Example: 'rent' -> 'RENT', 'Own' -> 'OWN'
            },
            'description': 'Standardize home ownership categories'
        },
        'loan_intent': {
            'mapping': {},
            'description': 'Standardize loan intent categories'
        },
        'loan_grade': {
            'mapping': {},
            'description': 'Standardize loan grade categories'
        },
        'historical_default': {
            'mapping': {
                'Y': 'Yes',
                'N': 'No',
                # NaN will be left as-is
            },
            'description': 'Standardize historical default to Yes/No'
        },
        'Current_loan_status': {
            'mapping': {},
            'description': 'Standardize loan status categories'
        }
    },
    
    # Binary/Boolean conversions
    'binary_conversions': {
        # Example: 'historical_default': {'Y': 1, 'N': 0}
        # If you want Y/N -> 1/0, configure here
    },
    
    # Numeric value cleanup
    'numeric_cleanup': {
        'loan_amnt': {
            'handle_zero': False,           # Whether to treat 0 as invalid
            'handle_negative': True,        # Remove negative values
            'description': 'Clean loan amounts'
        },
        'customer_income': {
            'handle_zero': False,
            'handle_negative': True,
            'description': 'Clean income values'
        }
    },
    
    # NaN handling (fill, remove, or flag)
    'nan_handling': {
        'loan_amnt': {
            'strategy': 'flag',             # 'remove', 'flag', or 'ignore'
            'indicator_name': 'loan_amnt_imputed',
            'description': 'Track missing loan amounts'
        },
        'historical_default': {
            'strategy': 'ignore',           # Leave NaN as-is for now
            'description': 'Leave missing default history'
        }
    }
}


# ==============================================================================
# ANALYSIS FUNCTION
# ==============================================================================

def analyze_values(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze values before correction.
    
    Identifies:
    - Unique categorical values and their frequencies
    - NaN patterns
    - Invalid numeric values (negative, zero where inappropriate)
    - Inconsistencies in categorical encoding (case, spacing)
    """
    analysis = {
        'categorical_analysis': {},
        'numeric_analysis': {},
        'nan_analysis': {},
        'issues_found': []
    }
    
    logger.info("\n" + "="*70)
    logger.info("VALUE ANALYSIS (BEFORE CORRECTION)")
    logger.info("="*70)
    
    # =========================================================================
    # CATEGORICAL ANALYSIS
    # =========================================================================
    logger.info("\nCategorical Columns:")
    logger.info(f"{'─'*70}")
    
    if 'categorical_standardization' in config:
        for col in config['categorical_standardization'].keys():
            if col not in df.columns:
                continue
            
            unique_vals = df[col].unique()
            value_counts = df[col].value_counts(dropna=False)
            
            logger.info(f"\n{col}:")
            logger.info(f"  Unique values: {len(unique_vals)}")
            logger.info(f"  Value distribution:")
            for val, count in value_counts.items():
                pct = (count / len(df)) * 100
                logger.info(f"    {str(val):20s}: {count:6d} ({pct:5.2f}%)")
            
            analysis['categorical_analysis'][col] = {
                'unique_count': len(unique_vals),
                'values': unique_vals.tolist(),
                'counts': value_counts.to_dict()
            }
    
    # =========================================================================
    # NUMERIC ANALYSIS
    # =========================================================================
    logger.info(f"\nNumeric Columns:")
    logger.info(f"{'─'*70}")
    
    if 'numeric_cleanup' in config:
        for col in config['numeric_cleanup'].keys():
            if col not in df.columns:
                continue
            
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"\n{col}: Not numeric (type: {df[col].dtype})")
                continue
            
            n_nan = df[col].isna().sum()
            n_zero = (df[col] == 0).sum()
            n_negative = (df[col] < 0).sum()
            
            logger.info(f"\n{col}:")
            logger.info(f"  NaN values: {n_nan}")
            logger.info(f"  Zero values: {n_zero}")
            logger.info(f"  Negative values: {n_negative}")
            logger.info(f"  Range: [{df[col].min():.2f}, {df[col].max():.2f}]")
            
            if n_negative > 0:
                analysis['issues_found'].append(f"{col}: {n_negative} negative values")
            
            analysis['numeric_analysis'][col] = {
                'nan_count': int(n_nan),
                'zero_count': int(n_zero),
                'negative_count': int(n_negative),
                'min': float(df[col].min()),
                'max': float(df[col].max())
            }
    
    logger.info(f"\n" + "="*70)
    return analysis


# ==============================================================================
# CORRECTION FUNCTION
# ==============================================================================

def correct_values(
    df: pd.DataFrame,
    config: Dict[str, Any] = None,
    analysis: Dict[str, Any] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Correct and standardize values across all columns.
    
    Handles:
    - Categorical value standardization (case, spacing, inconsistencies)
    - Binary conversions (Y/N -> 1/0, etc.)
    - Numeric cleanup (negative values, zeros)
    - NaN handling (remove, flag, or ignore)
    - Comprehensive audit logging
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe (post-type-correction)
    config : Dict
        Value correction configuration
    analysis : Dict, optional
        Pre-computed value analysis
    
    Returns:
    --------
    df_clean : pd.DataFrame
        Corrected dataframe
    audit : Dict
        Comprehensive audit trail
    """
    if config is None:
        config = VALUE_CORRECTION_CONFIG
    
    if analysis is None:
        analysis = analyze_values(df, config)
    
    audit = {
        'status': 'STARTED',
        'rows_input': len(df),
        'rows_output': len(df),
        'corrections_made': [],
        'rows_removed': 0,
        'columns_modified': [],
        'warnings': [],
        'errors': [],
        'details': {}
    }
    
    df_clean = df.copy()
    
    logger.info("\n" + "="*70)
    logger.info("VALUE CORRECTION EXECUTION")
    logger.info("="*70)
    
    # =========================================================================
    # STEP 1: CATEGORICAL STANDARDIZATION
    # =========================================================================
    logger.info("\nStep 1: Categorical value standardization")
    logger.info(f"{'─'*70}")
    
    if 'categorical_standardization' in config:
        for col, col_config in config['categorical_standardization'].items():
            if col not in df_clean.columns:
                continue
            
            mapping = col_config.get('mapping', {})
            
            if not mapping:
                logger.info(f"  {col:30s}: No standardization needed")
                continue
            
            # Apply mapping
            original_values = df_clean[col].copy()
            df_clean[col] = df_clean[col].map(mapping).fillna(df_clean[col])
            
            n_changed = (original_values != df_clean[col]).sum()
            
            if n_changed > 0:
                logger.info(f"  {col:30s}: Standardized {n_changed} values")
                audit['corrections_made'].append(col)
                audit['columns_modified'].append(col)
                audit['details'][col] = {'standardized': int(n_changed)}
            else:
                logger.info(f"  {col:30s}: No changes needed")
    
    # =========================================================================
    # STEP 2: HANDLE NaN IN NUMERIC COLUMNS
    # =========================================================================
    logger.info(f"\nStep 2: Handling NaN values")
    logger.info(f"{'─'*70}")
    
    if 'nan_handling' in config:
        for col, col_config in config['nan_handling'].items():
            if col not in df_clean.columns:
                continue
            
            n_nan = df_clean[col].isna().sum()
            if n_nan == 0:
                logger.info(f"  {col:30s}: No NaN values")
                continue
            
            strategy = col_config.get('strategy', 'ignore')
            
            if strategy == 'remove':
                rows_before = len(df_clean)
                df_clean = df_clean.dropna(subset=[col])
                rows_removed = rows_before - len(df_clean)
                audit['rows_removed'] += rows_removed
                logger.info(f"  {col:30s}: Removed {rows_removed} rows with NaN")
            
            elif strategy == 'flag':
                indicator_name = col_config.get('indicator_name', f'{col}_missing')
                df_clean[indicator_name] = df_clean[col].isna().astype(int)
                logger.info(f"  {col:30s}: Created indicator '{indicator_name}' ({n_nan} flagged)")
            
            elif strategy == 'ignore':
                logger.info(f"  {col:30s}: Keeping {n_nan} NaN values (ignored)")
    
    # =========================================================================
    # STEP 3: NUMERIC CLEANUP
    # =========================================================================
    logger.info(f"\nStep 3: Numeric value cleanup")
    logger.info(f"{'─'*70}")
    
    if 'numeric_cleanup' in config:
        for col, col_config in config['numeric_cleanup'].items():
            if col not in df_clean.columns:
                continue
            
            if not pd.api.types.is_numeric_dtype(df_clean[col]):
                logger.warning(f"  {col:30s}: Not numeric; skipping")
                continue
            
            rows_before = len(df_clean)
            handle_negative = col_config.get('handle_negative', False)
            
            if handle_negative:
                n_negative = (df_clean[col] < 0).sum()
                if n_negative > 0:
                    df_clean = df_clean[df_clean[col] >= 0]
                    rows_removed = rows_before - len(df_clean)
                    audit['rows_removed'] += rows_removed
                    logger.info(f"  {col:30s}: Removed {rows_removed} rows with negative values")
                else:
                    logger.info(f"  {col:30s}: No negative values found")
            else:
                logger.info(f"  {col:30s}: Negative values retained")
    
    # =========================================================================
    # STEP 4: FINAL SUMMARY
    # =========================================================================
    audit['rows_output'] = len(df_clean)
    
    logger.info(f"\n" + "="*70)
    logger.info("VALUE CORRECTION SUMMARY")
    logger.info("="*70)
    logger.info(f"Status: SUCCESS")
    logger.info(f"Corrections made: {len(audit['corrections_made'])}")
    if audit['corrections_made']:
        logger.info(f"  → {', '.join(audit['corrections_made'])}")
    logger.info(f"Columns modified: {len(audit['columns_modified'])}")
    logger.info(f"Rows removed: {audit['rows_removed']}")
    logger.info(f"Rows: {audit['rows_input']} → {audit['rows_output']} (removed: {audit['rows_removed']})")
    logger.info(f"Warnings: {len(audit['warnings'])}")
    logger.info(f"="*70)
    
    audit['status'] = 'SUCCESS'
    return df_clean, audit


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    # After Section 3.4 (Data Type Corrections):
    
    # Step 1: Analyze values
    value_analysis = analyze_values(data_typed, VALUE_CORRECTION_CONFIG)
    
    # Step 2: Correct values
    data_corrected, value_audit = correct_values(data_typed, VALUE_CORRECTION_CONFIG, value_analysis)
    
    # Step 3: Continue to Section 3.6 (Validation)
    print(f"\nData ready for final validation")
    print(f"Rows: {data_corrected.shape[0]}")
    print(f"Columns: {data_corrected.shape[1]}")
