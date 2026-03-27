# REF: Handle Missing Values
*Cookbook: `02_data_cleaning/`*

Ad-hoc imputation snippets. Pick the strategy that fits.

📋 See `__REF_DATA_CLEANING_MASTER.ipynb` Step 3 for the decision matrix.  
📋 See `TMPL_PIPELINE_missing_values.ipynb` for the audit-logged pipeline version.

### 1. The Comprehensive Null Audit

Standard .isnull() misses string-based placeholders. This function identifies them so you can treat them as true NaN values before cleaning begins.


```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns

def null_diagnostic(df):
    """Detects standard and 'hidden' nulls (e.g., '?', 'n/a', 'None')."""
    hidden_patterns = ['?', 'n/a', 'nan', 'none', '', ' ', 'null']
    
    diagnostic = pd.DataFrame({
        'standard_nulls': df.isnull().sum(),
        'hidden_nulls':   df.apply(lambda x: x.astype(str).str.lower().strip().isin(hidden_patterns).sum()),
        'null_pct':       (df.isnull().mean() * 100).round(2),
        'dtype':          df.dtypes
    })
    
    # Filter only columns that have issues
    return diagnostic.query('standard_nulls > 0 or hidden_nulls > 0').sort_values('null_pct', ascending=False)

# Usage: display(null_diagnostic(df))
```

### 2. Strategic "Smart" Imputation
A global median can distort data. Using transform with a grouping column (like department or category) provides a much more localized and accurate fill.


```python
def smart_impute(df, target_col, group_col, strategy='median'):
    """Imputes missing values based on subgroup statistics."""
    before_nulls = df[target_col].isnull().sum()
    
    if strategy == 'median':
        df[target_col] = df.groupby(group_col)[target_col].transform(lambda x: x.fillna(x.median()))
    elif strategy == 'mode':
        df[target_col] = df.groupby(group_col)[target_col].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))
        
    after_nulls = df[target_col].isnull().sum()
    print(f"✅ {target_col}: Imputed {before_nulls - after_nulls} values using {group_col} {strategy}.")
    return df

# Usage: df = smart_impute(df, 'salary', 'job_title', strategy='median')
```

### 3. Missingness Signal (The "Flag" Pattern)
Crucial for Scikit-Learn pipelines: if a value is missing, that fact might be a feature in itself.


```python
def add_missing_indicators(df, threshold=0.05):
    """Adds binary columns flagging where data was originally missing."""
    cols_to_flag = df.columns[df.isnull().mean() > threshold]
    
    for col in cols_to_flag:
        df[f"{col}_is_missing"] = df[col].isnull().astype(int)
        
    print(f"🚩 Created {len(cols_to_flag)} indicator columns for fields with >{threshold*100}% missingness.")
    return df
```

| Situation                          | Recommended Action                                                                 | Modular Tool to Use                         |
|-----------------------------------|-------------------------------------------------------------------------------------|---------------------------------------------|
| Strings like '?', 'N/A', or 'None'| Convert to true NaN before any analysis                                             | `null_diagnostic()`                         |
| Numeric (<5% missing)             | Group-wise Median (e.g., fill salary by job title)                                 | `smart_impute(strategy='median')`           |
| Categorical (<5% missing)         | Group-wise Mode (e.g., fill city by zip code)                                      | `smart_impute(strategy='mode')`             |
| Informative Missingness           | Keep the "signal" by adding a binary indicator                                     | `add_missing_indicators()`                  |
| High Missingness (>40%)           | Drop column OR convert to binary is_present flag                                   | `df.drop()` or indicator                    |
| Time-Series / Sequential          | Forward-fill followed by backward-fill                                             | `ffill().bfill()`                           |
| Complex Correlations             | Multi-variate estimation (KNN or Iterative)                                        | `sklearn.impute.KNNImputer`                 |
