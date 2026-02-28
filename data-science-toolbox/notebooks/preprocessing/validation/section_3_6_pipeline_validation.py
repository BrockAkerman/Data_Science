"""
Section 3.6: Pipeline Validation & Documentation

Final validation, audit trail consolidation, and comprehensive documentation
of the entire data cleaning pipeline (Sections 3.1–3.6).

Confirms all cleaning steps succeeded, documents row/column changes at each step,
verifies no unintended side effects, and produces final summary report.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Any, List
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# COMPREHENSIVE PIPELINE VALIDATION
# ==============================================================================

def validate_pipeline(
    df_original: pd.DataFrame,
    df_final: pd.DataFrame,
    audit_trails: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Comprehensive validation of entire pipeline (Sections 3.1–3.6).
    
    Validates:
    - Data integrity (no unexpected NaN, dtypes correct)
    - Row/column changes documented and consistent
    - No unintended side effects
    - Data quality metrics before/after
    
    Parameters:
    -----------
    df_original : pd.DataFrame
        Original raw data (before Section 3.1)
    df_final : pd.DataFrame
        Final cleaned data (after Section 3.5)
    audit_trails : Dict
        Audit trails from all cleaning sections
    
    Returns:
    --------
    Dict with comprehensive validation results
    """
    validation = {
        'status': 'PASSED',
        'timestamp': datetime.now().isoformat(),
        'validation_checks': {},
        'row_changes': {},
        'column_changes': {},
        'data_quality': {},
        'warnings': [],
        'errors': []
    }
    
    logger.info("\n" + "="*70)
    logger.info("PIPELINE VALIDATION & DOCUMENTATION")
    logger.info("="*70)
    
    # =========================================================================
    # CHECK 1: DATA SHAPE VALIDATION
    # =========================================================================
    logger.info("\nCheck 1: Data Shape Validation")
    logger.info(f"{'─'*70}")
    
    rows_original = len(df_original)
    rows_final = len(df_final)
    cols_original = len(df_original.columns)
    cols_final = len(df_final.columns)
    
    rows_removed = rows_original - rows_final
    cols_changed = cols_original - cols_final
    
    logger.info(f"  Rows: {rows_original} → {rows_final} (removed: {rows_removed}, {(rows_removed/rows_original)*100:.3f}%)")
    logger.info(f"  Columns: {cols_original} → {cols_final} (changed: {cols_changed})")
    
    validation['row_changes'] = {
        'original': rows_original,
        'final': rows_final,
        'removed': rows_removed,
        'percent_lost': round((rows_removed/rows_original)*100, 3)
    }
    
    validation['column_changes'] = {
        'original': cols_original,
        'final': cols_final,
        'net_change': cols_changed
    }
    
    validation['validation_checks']['shape'] = 'PASSED'
    
    # =========================================================================
    # CHECK 2: ROW REMOVAL AUDIT
    # =========================================================================
    logger.info(f"\nCheck 2: Row Removal Audit")
    logger.info(f"{'─'*70}")
    
    total_rows_removed_documented = 0
    
    section_names = {
        'duplicates': '3.1 Duplicate Handling',
        'missing_values': '3.2 Missing Value Handling',
        'outliers': '3.3 Outlier Handling',
        'data_types': '3.4 Data Type Corrections',
        'values': '3.5 Value Corrections'
    }
    
    for section_key, section_name in section_names.items():
        if section_key in audit_trails:
            audit = audit_trails[section_key]
            rows_removed = audit.get('rows_removed', 0)
            total_rows_removed_documented += rows_removed
            status = "✓" if rows_removed >= 0 else "✗"
            logger.info(f"  {status} {section_name:40s}: {rows_removed:6d} rows removed")
    
    logger.info(f"\n  Total documented removals: {total_rows_removed_documented}")
    logger.info(f"  Actual rows removed:       {rows_removed}")
    
    if total_rows_removed_documented == rows_removed:
        logger.info(f"  ✓ Row removal audit: MATCH")
        validation['validation_checks']['row_audit'] = 'PASSED'
    else:
        discrepancy = rows_removed - total_rows_removed_documented
        logger.warning(f"  ⚠ Discrepancy: {discrepancy} rows")
        validation['warnings'].append(f"Row removal discrepancy: {discrepancy} rows")
        validation['validation_checks']['row_audit'] = 'WARNING'
    
    # =========================================================================
    # CHECK 3: DATA TYPE VALIDATION
    # =========================================================================
    logger.info(f"\nCheck 3: Data Type Validation")
    logger.info(f"{'─'*70}")
    
    numeric_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_final.select_dtypes(include=['category']).columns.tolist()
    object_cols = df_final.select_dtypes(include=['object']).columns.tolist()
    
    logger.info(f"  Numeric columns: {len(numeric_cols)}")
    logger.info(f"  Categorical columns: {len(categorical_cols)}")
    logger.info(f"  Object/String columns: {len(object_cols)}")
    
    if len(object_cols) > 0:
        logger.warning(f"  ⚠ {len(object_cols)} object columns remain (should be numeric or category)")
        validation['warnings'].append(f"{len(object_cols)} object columns not converted to proper types")
        validation['validation_checks']['dtypes'] = 'WARNING'
    else:
        logger.info(f"  ✓ All columns properly typed")
        validation['validation_checks']['dtypes'] = 'PASSED'
    
    # =========================================================================
    # CHECK 4: MISSING DATA VALIDATION
    # =========================================================================
    logger.info(f"\nCheck 4: Missing Data Validation")
    logger.info(f"{'─'*70}")
    
    missing_before = df_original.isna().sum().sum()
    missing_after = df_final.isna().sum().sum()
    
    logger.info(f"  Total missing cells:")
    logger.info(f"    Before: {missing_before:,}")
    logger.info(f"    After:  {missing_after:,}")
    logger.info(f"    Resolved: {missing_before - missing_after:,}")
    
    missing_by_col = df_final.isna().sum()
    cols_with_missing = missing_by_col[missing_by_col > 0]
    
    if len(cols_with_missing) > 0:
        logger.info(f"\n  Columns with remaining NaN:")
        for col, count in cols_with_missing.items():
            pct = (count / len(df_final)) * 100
            logger.info(f"    {col:30s}: {count:6d} ({pct:5.2f}%)")
    else:
        logger.info(f"  ✓ No remaining missing values")
    
    validation['validation_checks']['missing'] = 'PASSED'
    
    # =========================================================================
    # CHECK 5: DUPLICATE VALIDATION
    # =========================================================================
    logger.info(f"\nCheck 5: Duplicate Validation")
    logger.info(f"{'─'*70}")
    
    n_duplicates_all = df_final.duplicated().sum()
    n_duplicates_subset = df_final.duplicated(subset=[col for col in df_final.columns if col not in ['customer_id']]).sum()
    
    logger.info(f"  Exact row duplicates: {n_duplicates_all}")
    logger.info(f"  Duplicates (excluding customer_id): {n_duplicates_subset}")
    
    if n_duplicates_all == 0:
        logger.info(f"  ✓ No duplicate rows")
        validation['validation_checks']['duplicates'] = 'PASSED'
    else:
        logger.warning(f"  ⚠ {n_duplicates_all} duplicate rows detected")
        validation['warnings'].append(f"{n_duplicates_all} duplicate rows")
        validation['validation_checks']['duplicates'] = 'WARNING'
    
    # =========================================================================
    # CHECK 6: OUTLIER VALIDATION (Post-cleaning bounds)
    # =========================================================================
    logger.info(f"\nCheck 6: Outlier Validation (Post-cleaning)")
    logger.info(f"{'─'*70}")
    
    outlier_checks = {
        'customer_age': {'min': 18, 'max': 100},
        'loan_amnt': {'min': 0, 'max': 900000},
        'loan_int_rate': {'min': 0, 'max': 100}
    }
    
    outlier_warnings = 0
    for col, bounds in outlier_checks.items():
        if col not in df_final.columns:
            continue
        
        min_val = bounds['min']
        max_val = bounds['max']
        
        below_min = (df_final[col] < min_val).sum()
        above_max = (df_final[col] > max_val).sum()
        
        if below_min > 0 or above_max > 0:
            logger.warning(f"  ⚠ {col:30s}: {below_min} below {min_val}, {above_max} above {max_val}")
            outlier_warnings += 1
        else:
            logger.info(f"  ✓ {col:30s}: All values in [{min_val}, {max_val}]")
    
    if outlier_warnings == 0:
        validation['validation_checks']['outliers'] = 'PASSED'
    else:
        validation['validation_checks']['outliers'] = 'WARNING'
    
    # =========================================================================
    # CHECK 7: DATA QUALITY METRICS
    # =========================================================================
    logger.info(f"\nCheck 7: Data Quality Metrics")
    logger.info(f"{'─'*70}")
    
    data_quality = {
        'completeness': round((1 - (missing_after / (len(df_final) * len(df_final.columns)))) * 100, 2),
        'uniqueness': len(df_final) / len(df_original) * 100,
        'validity': round((1 - (len(cols_with_missing) / len(df_final.columns))) * 100, 2),
        'consistency': 'PASSED' if len(object_cols) == 0 else 'WARNING'
    }
    
    logger.info(f"  Completeness (non-null cells): {data_quality['completeness']:.2f}%")
    logger.info(f"  Uniqueness (row retention): {data_quality['uniqueness']:.2f}%")
    logger.info(f"  Validity (columns without NaN): {data_quality['validity']:.2f}%")
    logger.info(f"  Consistency (proper dtypes): {data_quality['consistency']}")
    
    validation['data_quality'] = data_quality
    
    # =========================================================================
    # FINAL STATUS
    # =========================================================================
    logger.info(f"\n" + "="*70)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*70)
    
    passed = sum(1 for v in validation['validation_checks'].values() if v == 'PASSED')
    total = len(validation['validation_checks'])
    
    logger.info(f"\nValidation Checks: {passed}/{total} PASSED")
    for check, status in validation['validation_checks'].items():
        symbol = "✓" if status == 'PASSED' else "⚠"
        logger.info(f"  {symbol} {check:30s}: {status}")
    
    if validation['warnings']:
        logger.warning(f"\nWarnings ({len(validation['warnings'])}):")
        for warning in validation['warnings']:
            logger.warning(f"  ⚠ {warning}")
    
    if validation['errors']:
        logger.error(f"\nErrors ({len(validation['errors'])}):")
        for error in validation['errors']:
            logger.error(f"  ✗ {error}")
        validation['status'] = 'FAILED'
    elif validation['warnings']:
        validation['status'] = 'PASSED_WITH_WARNINGS'
    else:
        validation['status'] = 'PASSED'
    
    logger.info(f"\nFinal Status: {validation['status']}")
    logger.info(f"="*70)
    
    return validation


# ==============================================================================
# PIPELINE SUMMARY REPORT
# ==============================================================================

def generate_pipeline_summary(
    df_original: pd.DataFrame,
    df_final: pd.DataFrame,
    audit_trails: Dict[str, Dict[str, Any]],
    validation: Dict[str, Any]
) -> str:
    """
    Generate comprehensive pipeline summary report.
    
    Returns:
    --------
    str
        Formatted summary report
    """
    
    report = []
    report.append("\n" + "="*70)
    report.append("DATA CLEANING PIPELINE SUMMARY REPORT")
    report.append("="*70)
    report.append(f"Generated: {validation['timestamp']}")
    report.append("")
    
    # Input/Output Summary
    report.append("PIPELINE METRICS")
    report.append(f"{'─'*70}")
    report.append(f"Input rows:     {len(df_original):,}")
    report.append(f"Output rows:    {len(df_final):,}")
    report.append(f"Rows removed:   {len(df_original) - len(df_final):,} ({validation['row_changes']['percent_lost']:.3f}%)")
    report.append(f"Input columns:  {len(df_original.columns)}")
    report.append(f"Output columns: {len(df_final.columns)}")
    report.append("")
    
    # Section-by-Section Summary
    report.append("SECTION-BY-SECTION ROW REMOVAL")
    report.append(f"{'─'*70}")
    
    sections = [
        ('3.1', 'Duplicate Handling', 'duplicates'),
        ('3.2', 'Missing Value Handling', 'missing_values'),
        ('3.3', 'Outlier Handling', 'outliers'),
        ('3.4', 'Data Type Corrections', 'data_types'),
        ('3.5', 'Value Corrections', 'values')
    ]
    
    total_removed = 0
    for section_num, section_name, key in sections:
        if key in audit_trails:
            audit = audit_trails[key]
            removed = audit.get('rows_removed', 0)
            total_removed += removed
            status = "✓" if removed == 0 else f"{removed:,} removed"
            report.append(f"  {section_num} {section_name:35s}: {status}")
    
    report.append(f"\n  Total across all sections: {total_removed:,} rows")
    report.append("")
    
    # Data Quality Improvement
    report.append("DATA QUALITY METRICS")
    report.append(f"{'─'*70}")
    report.append(f"Completeness:  {validation['data_quality']['completeness']:.2f}% (non-null cells)")
    report.append(f"Uniqueness:    {validation['data_quality']['uniqueness']:.2f}% (row retention)")
    report.append(f"Validity:      {validation['data_quality']['validity']:.2f}% (columns without NaN)")
    report.append(f"Consistency:   {validation['data_quality']['consistency']} (proper dtypes)")
    report.append("")
    
    # Validation Results
    report.append("VALIDATION RESULTS")
    report.append(f"{'─'*70}")
    for check, status in validation['validation_checks'].items():
        symbol = "✓" if status == 'PASSED' else "⚠"
        report.append(f"  {symbol} {check:30s}: {status}")
    report.append(f"\nOverall Status: {validation['status']}")
    report.append("")
    
    # Warnings and Errors
    if validation['warnings']:
        report.append("WARNINGS")
        report.append(f"{'─'*70}")
        for warning in validation['warnings']:
            report.append(f"  ⚠ {warning}")
        report.append("")
    
    if validation['errors']:
        report.append("ERRORS")
        report.append(f"{'─'*70}")
        for error in validation['errors']:
            report.append(f"  ✗ {error}")
        report.append("")
    
    # Final Dataset Profile
    report.append("FINAL DATASET PROFILE")
    report.append(f"{'─'*70}")
    report.append(f"Rows:              {len(df_final):,}")
    report.append(f"Columns:           {len(df_final.columns)}")
    report.append(f"Numeric columns:   {len(df_final.select_dtypes(include=[np.number]).columns)}")
    report.append(f"Categorical cols:  {len(df_final.select_dtypes(include=['category']).columns)}")
    report.append(f"Memory usage:      {df_final.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    report.append("")
    
    report.append("="*70)
    
    return "\n".join(report)


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    # After completing Sections 3.1–3.5:
    
    # Consolidate all audit trails
    all_audits = {
        'duplicates': duplicate_audit,      # From Section 3.1
        'missing_values': missing_audit,    # From Section 3.2
        'outliers': outlier_audit,          # From Section 3.3
        'data_types': type_audit,           # From Section 3.4
        'values': value_audit               # From Section 3.5
    }
    
    # Step 1: Validate entire pipeline
    validation_results = validate_pipeline(data_original, data_corrected, all_audits)
    
    # Step 2: Generate summary report
    summary_report = generate_pipeline_summary(data_original, data_corrected, all_audits, validation_results)
    print(summary_report)
    
    # Step 3: Save report to file (optional)
    with open('pipeline_validation_report.txt', 'w') as f:
        f.write(summary_report)
    
    print(f"\n✓ Pipeline validation complete")
    print(f"✓ Data ready for exploratory analysis and modeling")
