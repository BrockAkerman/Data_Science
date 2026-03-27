# 🧹 Template: Data Cleaning Pipeline

**Purpose:** Reusable audit-trail-based cleaning pipeline.  
**Components:**
1. Duplicate Handling
2. Missing Value Handling
3. Data Type Standardization
4. Outlier Detection & Treatment
5. Value Correction

**How to use:**  
- Update the CONFIG dicts to match your dataset's column names and strategies.  
- Run the cells top-to-bottom. Each step produces an `audit` dict you can inspect.


### 1. The Global Pipeline Config
By keeping the configuration separate from the logic, you can reuse the code below for every project and only ever change this single cell.


```python
# ==============================================================================
# PIPELINE CONFIGURATION (Template)
# ==============================================================================
# Replace 'col_name' with your actual columns.
CLEANING_CONFIG = {
    'duplicates': {
        'active': True,
        'subset': None,            # List of cols or None for all
        'id_col': 'unique_id',     # Primary key to check for entity conflicts
        'sort_by': 'timestamp',    # Keep latest record based on this col
        'keep': 'first',           # 'first', 'last', or False
        'fail_if_remain': True     # Raise error if duplicates persist
    },
    'missing_values': {
        # Feature-specific strategies
        'numeric_feature': {'strategy': 'median', 'group_by': 'category_col'},
        'cat_feature':     {'strategy': 'mode',   'group_by': 'category_col'},
        'sparse_feature':  {'strategy': 'indicator_only'}, 
    },
    'outliers': {
        'skewed_feature':  {'strategy': 'winsorize', 'limits': (0.01, 0.01)},
        'growth_feature':  {'strategy': 'log_transform'},
    },
    'standardization': {
        'snake_case': True,
        'trim_whitespace': True
    }
}
```

### 2. The Modular Cleaning Engine
Each function here is designed to be independent. You can copy just the "Duplicate" function or the "Outlier" function as needed.


```python
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def pipeline_duplicates(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """
    Handles exact/entity duplicates with sorting logic and safety audits.
    """
    if not cfg.get('active'): 
        return df, {'status': 'SKIPPED'}
    
    audit = {
        'status': 'STARTED', 
        'rows_input': len(df), 
        'checks': []
    }
    
    # 1. Sort to ensure 'keep' targets the correct record
    if cfg.get('sort_by') and cfg.get('sort_by') in df.columns:
        df = df.sort_values(by=cfg['sort_by'], ascending=False)
        audit['checks'].append(f"Sorted by {cfg['sort_by']}")

    # 2. Audit exact duplicates before dropping
    exact_dupes = df.duplicated(subset=cfg.get('subset')).sum()
    
    # 3. Drop
    df = df.drop_duplicates(subset=cfg.get('subset'), keep=cfg.get('keep'))
    
    # 4. Safety Check: Verify uniqueness of ID column
    id_col = cfg.get('id_col')
    if id_col and id_col in df.columns:
        remaining_id_dupes = df.duplicated(subset=[id_col]).sum()
        if remaining_id_dupes > 0:
            msg = f"❌ DUPLICATES: {remaining_id_dupes} ID conflicts remain in {id_col}."
            if cfg.get('fail_if_remain'):
                raise ValueError(msg)
            logger.warning(msg)

    audit.update({
        'status': 'SUCCESS',
        'rows_removed': audit['rows_input'] - len(df),
        'exact_dupes_found': exact_dupes
    })
    
    logger.info(f"✂️ DUPLICATES: Removed {audit['rows_removed']} rows.")
    return df, audit

def pipeline_missing_values(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """
    Imputes missing values using subgroup-aware logic and tracks the audit trail.
    """
    audit = {'status': 'STARTED', 'imputations': {}, 'indicators': []}
    
    # Standardize 'hidden' nulls first
    hidden_nulls = ['?', 'n/a', 'nan', 'none', '', 'null']
    df = df.replace(hidden_nulls, np.nan)
    
    for col, settings in cfg.items():
        if col not in df.columns:
            logger.warning(f"⚠️ MISSING: Column {col} not found. Skipping.")
            continue
        
        strategy = settings.get('strategy')
        group = settings.get('group_by')
        before_nulls = int(df[col].isnull().sum())
        
        if strategy == 'indicator_only':
            df[f'{col}_is_missing'] = df[col].isnull().astype(int)
            audit['indicators'].append(col)
            
        elif strategy in ['median', 'mode'] and group:
            if group not in df.columns:
                logger.error(f"❌ Group col {group} missing for {col} imputation.")
                continue
                
            if strategy == 'median':
                df[col] = df.groupby(group)[col].transform(lambda x: x.fillna(x.median()))
            else:
                df[col] = df.groupby(group)[col].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))
            
            after_nulls = int(df[col].isnull().sum())
            audit['imputations'][col] = {'removed': before_nulls - after_nulls, 'remaining': after_nulls}
            logger.info(f"🩹 MISSING: {col} imputed via {group} ({strategy}).")
            
    audit['status'] = 'SUCCESS'
    return df, audit

def pipeline_outliers(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """
    Treats outliers and logs the specific strategies applied for documentation.
    """
    from scipy.stats.mstats import winsorize
    audit = {'status': 'STARTED', 'treatments': []}
    
    for col, settings in cfg.items():
        if col not in df.columns: continue
        
        strategy = settings.get('strategy')
        if strategy == 'winsorize':
            limits = settings.get('limits', (0.01, 0.01))
            df[col] = winsorize(df[col], limits=limits)
            audit['treatments'].append({'col': col, 'strategy': 'winsorize', 'limits': limits})
            logger.info(f"📈 OUTLIERS: Winsorized {col} at {limits}.")
            
        elif strategy == 'log_transform':
            # Ensure no negative values before log
            if (df[col] < 0).any():
                logger.warning(f"⚠️ OUTLIERS: {col} contains negative values. Log1p may produce NaNs.")
            df[col] = np.log1p(df[col])
            audit['treatments'].append({'col': col, 'strategy': 'log_transform'})
            logger.info(f"📉 OUTLIERS: Log-transformed {col}.")
            
    audit['status'] = 'SUCCESS'
    return df, audit

def pipeline_standardization(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """
    Standardizes schema and formats with error handling for mixed types.
    """
    audit = {'status': 'STARTED', 'actions': []}
    
    if cfg.get('snake_case'):
        df.columns = [str(c).lower().replace(' ', '_').strip() for c in df.columns]
        audit['actions'].append('snake_case_columns')
        
    if cfg.get('trim_whitespace'):
        # Only target columns that are actually string/object types
        str_cols = df.select_dtypes(['object', 'string']).columns
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip()
        audit['actions'].append('trimmed_whitespace')
        
    logger.info("🔡 FORMATTING: Schema and text standardized.")
    audit['status'] = 'SUCCESS'
    return df, audit
```

### 3. The Execution Cell
This is your "Plug-and-Play" cell. You drop this at the top of your cleaning section.


```python
def run_cleaning_pipeline(df_input: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Orchestrates the modular cleaning functions in order."""
    df = df_input.copy()
    
    df = pipeline_standardization(df, config.get('standardization', {}))
    df = pipeline_duplicates(df, config.get('duplicates', {}))
    df = pipeline_missing_values(df, config.get('missing_values', {}))
    df = pipeline_outliers(df, config.get('outliers', {}))
    
    logger.info("✨ PIPELINE COMPLETE: Data is clean and ready for analysis.")
    return df

# Final usage:
# df = run_cleaning_pipeline(raw_df, CLEANING_CONFIG)
```
