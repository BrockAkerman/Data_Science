# REF: Duplicate Handling
*Cookbook: `02_data_cleaning/`*

Ad-hoc snippets for finding and removing duplicates. Grab what you need.

📋 See `__REF_DATA_CLEANING_MASTER.ipynb` Step 2 for decision guidance.  
📋 See `TMPL_PIPELINE_duplicate_handling.ipynb` for audit-logged pipeline version.

### 1. The Duplicate Audit

This function distinguishes between "Exact Duplicates" (useless data) and "ID Conflicts" (potentially dangerous data).


```python
def duplicate_audit(df, id_col=None):
    """
    Analyzes exact duplicates and ID-based conflicts.
    
    Args:
        df: The dataframe to audit.
        id_col: The primary key or ID column to check for entity conflicts.
    """
    results = {
        'total_rows': len(df),
        'exact_duplicates': df.duplicated().sum(),
    }
    
    if id_col and id_col in df.columns:
        results['id_conflicts'] = df.duplicated(subset=[id_col]).sum()
        results['unique_ids'] = df[id_col].nunique()
        
    audit_report = pd.Series(results)
    print("📋 Duplicate Audit Report:")
    print(audit_report)
    return audit_report

# Usage: audit = duplicate_audit(df, id_col='user_id')
```

### 2. Strategic Duplicate Removal

Instead of a blind drop_duplicates(), this approach allows you to sort the data first (e.g., by a timestamp or "completeness") to ensure the best record is the one that stays.


```python
def clean_duplicates(df, subset=None, sort_col=None, ascending=False, verbose=True):
    """
    Drops duplicates with control over which record is 'first'.
    
    Args:
        subset: List of columns to define a duplicate (usually the ID).
        sort_col: Column to sort by before dropping (e.g., 'updated_at').
        ascending: Direction of sort (False = keep latest/highest).
    """
    before = len(df)
    
    # 1. Sort to ensure 'keep=first' targets the right row
    if sort_col:
        df = df.sort_values(by=sort_col, ascending=ascending)
        
    # 2. Drop duplicates
    df = df.drop_duplicates(subset=subset, keep='first')
    
    after = len(df)
    if verbose:
        print(f"✂️ Removed {before - after} rows ({after} remaining).")
        
    return df

# Usage: Keep the most recent record for each user
# df = clean_duplicates(df, subset=['user_id'], sort_col='timestamp', ascending=False)
```

### 3. Verification & Safety Assertions

This ensures your "cleaning" didn't accidentally wipe out your entire dataset or leave logical errors behind.


```python
def verify_uniqueness(df, id_col):
    """Safety check for pipeline integration."""
    is_unique = df[id_col].is_unique
    if not is_unique:
        duplicates = df[df.duplicated(subset=[id_col], keep=False)]
        print(f"❌ Verification Failed: {id_col} still has duplicates.")
        return duplicates.sort_values(id_col)
    else:
        print(f"✅ Verification Passed: {id_col} is a valid primary key.")
        return None

# verify_uniqueness(df, 'id')
```
