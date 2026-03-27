# REF: Downtyping (Memory Reduction)
*Cookbook: `02_data_cleaning/`*

Downcast numeric columns to reduce DataFrame memory footprint.
Run after type correction, before saving to disk or handing off to modeling.

📋 See `utils/helpers.py` for the `reduce_mem_usage()` utility function.

### 1. The Comprehensive Optimizer

This function is designed to be a "set and forget" utility in your modular library.


```python
def optimize_memory(df: pd.DataFrame, category_threshold: float = 0.5, verbose: bool = True) -> pd.DataFrame:
    """
    Optimizes memory by downcasting numerics and converting low-cardinality objects to categories.
    
    Args:
        df: The dataframe to optimize.
        category_threshold: Max ratio of unique values to total rows to trigger 'category' conversion.
        verbose: Print memory reduction stats.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    # 1. Handle Numerics
    # Using pd.to_numeric(downcast=...) is safer than manual min/max range checks
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
        
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
        
    # 2. Handle Objects (The most impactful reduction)
    for col in df.select_dtypes(include=['object']).columns:
        num_unique = df[col].nunique()
        num_total = len(df)
        
        # If cardinality is low, 'category' saves massive space vs 'object'
        if (num_unique / num_total) < category_threshold:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem
        print(f"✅ Memory Optimization Complete")
        print(f"   {start_mem:.2f} MB → {end_mem:.2f} MB ({reduction:.1f}% reduction)")
        
    return df

# Usage: df = optimize_memory(df)
```

### 2. Integration & Compatibility Checks

Since you are using Scikit-Learn and llama_index, we must ensure downcasting doesn't break your downstream modular code.


```python
def validate_downcast(df_old, df_new):
    """Ensures no data loss occurred during optimization."""
    # Check for precision loss in floats (sum should be roughly equal)
    num_cols = df_new.select_dtypes(include=np.number).columns
    diffs = (df_old[num_cols].sum() - df_new[num_cols].sum()).abs().sum()
    
    if diffs > 1e-5:
        print("⚠️ Warning: Precision loss detected in numeric columns.")
    else:
        print("✔️ Data integrity verified.")

# validate_downcast(original_df, optimized_df)
```

### When to Downtype
- After all cleaning is complete (types are stable)
- Before saving to disk (parquet especially benefits)
- Before training large models (saves memory bandwidth)
- **Not before** outlier detection — float32 precision loss can shift IQR bounds slightly
