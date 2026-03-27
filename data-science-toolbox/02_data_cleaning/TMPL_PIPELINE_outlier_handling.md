# Section 3.3: Outlier Handling Pipeline

This notebook demonstrates a production-ready outlier detection and treatment pipeline with comprehensive audit logging.

**Objective:** Identify and treat outliers using statistical methods (IQR) combined with domain knowledge and visual diagnostics.

### 1. The Global Configuration
This configuration is designed for "Plug-and-Play" use. You can define specific strategies (IQR, Z-score, or Domain-based) for each numeric feature.


```python
# ==============================================================================
# OUTLIER HANDLING CONFIGURATION
# ==============================================================================
OUTLIER_CONFIG = {
    'active': True,
    'features': {
        'feature_name_1': {
            'strategy': 'cap',           # Options: 'cap' (Winsorize), 'drop', 'transform'
            'method': 'iqr',             # Options: 'iqr', 'zscore'
            'multiplier': 1.5            # 1.5 for IQR, 3.0 for Z-score
        },
        'feature_name_2': {
            'strategy': 'transform', 
            'method': 'log1p'
        },
        'feature_name_3': {
            'strategy': 'drop',
            'method': 'domain',
            'bounds': (0, 1000)          # Hard limits for domain-specific logic
        }
    },
    'fail_if_outliers_remain': False     # Set to True for high-strictness pipelines
}
```

### 2. The Robust Outlier Engine
This function replaces the previous class structure. It provides a detailed audit trail including "fences" (the calculated boundaries) and the number of values impacted.


```python
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def pipeline_outlier_handling(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict]:
    """
    Identifies and treats outliers using IQR, Z-score, or Domain logic.
    Standardized for modular 'Plug-and-Play' integration.
    """
    if not cfg.get('active'):
        return df, {'status': 'SKIPPED'}

    audit = {
        'status': 'STARTED',
        'treatments': {},
        'rows_removed': 0,
        'features_processed': []
    }
    
    df_clean = df.copy()

    for col, settings in cfg.get('features', {}).items():
        if col not in df_clean.columns:
            logger.warning(f"⚠️ OUTLIERS: Column '{col}' not found. Skipping.")
            continue
            
        strategy = settings.get('strategy')
        method = settings.get('method')
        multiplier = settings.get('multiplier', 1.5)
        
        # Determine Fences
        lower_f, upper_f = None, None
        
        if method == 'iqr':
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            lower_f, upper_f = q1 - (multiplier * iqr), q3 + (multiplier * iqr)
        elif method == 'zscore':
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            lower_f, upper_f = mean - (multiplier * std), mean + (multiplier * std)
        elif method == 'domain':
            lower_f, upper_f = settings.get('bounds', (df_clean[col].min(), df_clean[col].max()))

        # Execution
        outlier_mask = (df_clean[col] < lower_f) | (df_clean[col] > upper_f)
        outlier_count = int(outlier_mask.sum())
        
        if outlier_count == 0 and strategy != 'transform':
            continue

        if strategy == 'cap':
            df_clean[col] = df_clean[col].clip(lower=lower_f, upper=upper_f)
            logger.info(f"📈 OUTLIERS: Capped {outlier_count} values in '{col}' via {method}.")
            
        elif strategy == 'drop':
            df_clean = df_clean[~outlier_mask]
            audit['rows_removed'] += outlier_count
            logger.info(f"✂️ OUTLIERS: Dropped {outlier_count} rows based on '{col}' outliers.")
            
        elif strategy == 'transform':
            if method == 'log1p':
                df_clean[f'{col}_log'] = np.log1p(df_clean[col])
                logger.info(f"📉 OUTLIERS: Log-transformed '{col}'.")

        audit['treatments'][col] = {
            'method': method,
            'strategy': strategy,
            'count': outlier_count,
            'fences': (round(lower_f, 2), round(upper_f, 2))
        }
        audit['features_processed'].append(col)

    audit['status'] = 'SUCCESS'
    logger.info(f"✅ OUTLIERS COMPLETE: Processed {len(audit['features_processed'])} features.")
    return df_clean, audit
```
