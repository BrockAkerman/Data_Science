# Section 3.1: Duplicate Handling Pipeline

This notebook demonstrates a production-ready duplicate handling pipeline with comprehensive audit logging.

**Objective:** Identify and remove exact row duplicates and customer-level duplicates before feature engineering.

### 1. The Global Configuration
This is now entirely generic. You only need to define which column acts as your "Primary Key" (Entity ID) and which column determines the "Source of Truth" (Priority/Timestamp).


```python
# ==============================================================================
# DUPLICATE HANDLING CONFIGURATION
# ==============================================================================
DUPLICATE_CONFIG = {
    'active': True,
    'id_column': 'primary_id_col',      # Column that must be unique (e.g., UserID)
    'subset': None,                    # None = check all columns for exact duplicates
    'priority_col': 'timestamp_col',   # Use to decide which record to keep
    'keep_strategy': 'first',          # 'first' keeps highest value of priority_col
    'fail_if_duplicates_remain': True  # Raise error if ID-conflicts persist after cleaning
}
```

### 2. The Robust Duplicate Engine
This function replaces the previous class structure. It provides a deep audit of "Exact Row" vs "Entity Level" duplicates, which is a high-value talking point in technical interviews.


```python
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def pipeline_duplicate_handling(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict]:
    """
    Handles exact row duplicates and logical entity conflicts with a full audit trail.
    Standardized for modular 'Plug-and-Play' integration.
    """
    if not cfg.get('active'):
        return df, {'status': 'SKIPPED'}

    audit = {
        'status': 'STARTED',
        'rows_input': len(df),
        'exact_duplicates_found': 0,
        'entity_conflicts_found': 0,
        'checks_performed': []
    }
    
    df_clean = df.copy()

    # --- STEP 1: Exact Row Duplicate Audit ---
    exact_count = int(df_clean.duplicated(subset=cfg.get('subset'), keep=False).sum())
    audit['exact_duplicates_found'] = exact_count
    audit['checks_performed'].append('exact_row_check')
    logger.info(f"🔍 DUPLICATES: Found {exact_count} exact row duplicates.")

    # --- STEP 2: Entity-Level Audit (ID Conflicts) ---
    id_col = cfg.get('id_column')
    if id_col and id_col in df_clean.columns:
        entity_dupes = len(df_clean) - df_clean[id_col].nunique()
        audit['entity_conflicts_found'] = int(entity_dupes)
        audit['checks_performed'].append('entity_level_check')
        logger.info(f"🔍 DUPLICATES: Found {entity_dupes} entity-level (ID) conflicts.")

    # --- STEP 3: Logical Removal ---
    # Sort by priority (e.g., most recent timestamp) before dropping
    p_col = cfg.get('priority_col')
    if p_col and p_col in df_clean.columns:
        df_clean = df_clean.sort_values(by=p_col, ascending=False)
        audit['checks_performed'].append('priority_sort')

    df_clean = df_clean.drop_duplicates(
        subset=cfg.get('subset'), 
        keep=cfg.get('keep_strategy', 'first')
    )

    # --- STEP 4: Final Validation ---
    if id_col and id_col in df_clean.columns:
        remaining = df_clean.duplicated(subset=[id_col]).sum()
        if remaining > 0:
            audit['status'] = 'FAILED'
            msg = f"❌ DUPLICATES: {remaining} conflicts remain in '{id_col}' after cleaning."
            if cfg.get('fail_if_duplicates_remain'):
                raise ValueError(msg)
            logger.error(msg)
        else:
            audit['status'] = 'SUCCESS'
    
    audit['rows_output'] = len(df_clean)
    audit['rows_removed'] = audit['rows_input'] - audit['rows_output']
    
    logger.info(f"✅ DUPLICATES COMPLETE: {audit['rows_removed']} rows removed.")
    return df_clean, audit
```
