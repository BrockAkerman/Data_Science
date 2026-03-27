# Section 3.2: Missing Value Handling Pipeline

This notebook demonstrates a production-ready missing value handling pipeline with comprehensive audit logging.

**Objective:** Analyze missingness patterns and execute feature-specific imputation/deletion strategies.

### 1. The Global Configuration
This is now scrubbed of specific feature names. It uses generic placeholders so you can drop in your actual column names as needed.


```python
# ==============================================================================
# MISSING VALUE HANDLING CONFIGURATION
# ==============================================================================
MISSING_CONFIG = {
    'active': True,
    'features': {
        'numeric_feature': {'strategy': 'median', 'group_by': 'category_col'},
        'categorical_feature': {'strategy': 'mode', 'group_by': 'category_col'},
        'sparse_feature': {'strategy': 'indicator_only'},
        'strict_feature': {'strategy': 'drop_rows'}
    },
    'hidden_nulls': ['?', 'n/a', 'nan', 'none', '', 'null'],
    'fail_if_nulls_remain': False  # Set True for strict production requirements
}
```


```python
def analyze_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Diagnostic tool: Provides a summary of missingness and identifies 
    potential 'hidden' nulls in the dataset.
    """
    # 1. Standard Null Count
    null_counts = df.isnull().sum()
    null_pct = (df.isnull().sum() / len(df)) * 100
    
    # 2. Check for 'Hidden' Nulls (Common in production data)
    hidden_lookups = ['?', 'n/a', 'nan', 'none', '', 'null']
    hidden_counts = {}
    
    for col in df.select_dtypes(include=['object', 'category']).columns:
        h_count = df[col].astype(str).str.lower().str.strip().isin(hidden_lookups).sum()
        if h_count > 0:
            hidden_counts[col] = h_count
            
    # 3. Combine into Report
    report = pd.DataFrame({
        'missing_count': null_counts,
        'pct_missing': null_pct.round(2),
        'dtype': df.dtypes
    })
    
    report['hidden_nulls_detected'] = report.index.map(hidden_counts).fillna(0).astype(int)
    
    # Only return columns that have some form of missingness
    report = report[(report['missing_count'] > 0) | (report['hidden_nulls_detected'] > 0)]
    
    logger.info(f"📊 DIAGNOSTIC: {len(report)} columns identified with missingness.")
    return report.sort_values(by='pct_missing', ascending=False)
```

### 2. The Robust Missing Value Engine
This function replaces the previous class structure, providing a detailed breakdown of how many nulls were caught, how many were "hidden," and the final imputation counts.


```python
import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Any

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def pipeline_missing_value_handling(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict]:
    """
    Analyzes and treats missingness with hidden-null detection and subgroup-aware imputation.
    Standardized for modular 'Plug-and-Play' integration.
    """
    if not cfg.get('active'):
        return df, {'status': 'SKIPPED'}

    audit = {
        'status': 'STARTED',
        'initial_nulls': int(df.isnull().sum().sum()),
        'hidden_nulls_found': 0,
        'imputations': {},
        'indicators_created': [],
        'rows_removed': 0
    }
    
    df_clean = df.copy()

    # --- STEP 1: Hidden Null Standardization ---
    if cfg.get('hidden_nulls'):
        before_standard = df_clean.isnull().sum().sum()
        df_clean = df_clean.replace(cfg['hidden_nulls'], np.nan)
        audit['hidden_nulls_found'] = int(df_clean.isnull().sum().sum() - before_standard)
        logger.info(f"🔍 MISSING: Standardized {audit['hidden_nulls_found']} hidden nulls.")

    # --- STEP 2: Feature-Specific Treatment ---
    for col, settings in cfg.get('features', {}).items():
        if col not in df_clean.columns:
            logger.warning(f"⚠️ MISSING: Column '{col}' not found. Skipping.")
            continue
            
        strategy = settings.get('strategy')
        group = settings.get('group_by')
        before_col_nulls = int(df_clean[col].isnull().sum())
        
        if before_col_nulls == 0 and strategy != 'indicator_only':
            continue

        # Strategy: Missing Indicator (Flagging)
        if strategy == 'indicator_only':
            df_clean[f'{col}_is_missing'] = df_clean[col].isnull().astype(int)
            audit['indicators_created'].append(col)
            logger.info(f"🚩 MISSING: Created indicator for '{col}'.")

        # Strategy: Drop Rows
        elif strategy == 'drop_rows':
            df_clean = df_clean.dropna(subset=[col])
            removed = before_col_nulls
            audit['rows_removed'] += removed
            logger.info(f"✂️ MISSING: Dropped {removed} rows due to nulls in '{col}'.")

        # Strategy: Subgroup Imputation (Median/Mode)
        elif strategy in ['median', 'mode'] and group:
            if group not in df_clean.columns:
                logger.error(f"❌ MISSING: Group col '{group}' missing for '{col}' imputation.")
                continue
            
            if strategy == 'median':
                df_clean[col] = df_clean.groupby(group)[col].transform(lambda x: x.fillna(x.median()))
            else:
                df_clean[col] = df_clean.groupby(group)[col].transform(
                    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan)
                )
            
            after_col_nulls = int(df_clean[col].isnull().sum())
            audit['imputations'][col] = {'count': before_col_nulls - after_col_nulls, 'strategy': f"{strategy} via {group}"}
            logger.info(f"🩹 MISSING: Imputed {audit['imputations'][col]['count']} values in '{col}'.")

    # --- STEP 3: Final Validation ---
    final_nulls = int(df_clean.isnull().sum().sum())
    audit['final_null_count'] = final_nulls
    
    if final_nulls > 0 and cfg.get('fail_if_nulls_remain'):
        audit['status'] = 'FAILED'
        raise ValueError(f"❌ MISSING: {final_nulls} nulls remain after pipeline execution.")
    
    audit['status'] = 'SUCCESS'
    logger.info(f"✅ MISSING COMPLETE: {audit['final_null_count']} total nulls remaining.")
    return df_clean, audit
```
