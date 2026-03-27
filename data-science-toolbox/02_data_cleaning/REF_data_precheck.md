# REF: Data Precheck
*Cookbook: `02_data_cleaning/`*

Run this before any cleaning. Understand the data before touching it.

📋 See `__REF_DATA_CLEANING_MASTER.ipynb` Step 1 for decision guidance.

### 1. Global Setup & Diagnostic Tools
Instead of just df.info(), we use a more robust audit function that catches edge cases.


```python
def diagnostic_audit(df):
    """Comprehensive health check of the dataframe structure."""
    audit = pd.DataFrame({
        'dtype': df.dtypes,
        'nulls': df.isnull().sum(),
        'null_pct': (df.isnull().mean() * 100).round(2),
        'unique': df.nunique(),
        'cardinality_pct': (df.nunique() / len(df) * 100).round(2),
        'zeros': (df == 0).sum() if df.select_dtypes(include=np.number).shape[1] > 0 else 0,
        'constant': df.nunique() <= 1
    })
    
    # Identify 'Hidden' Nulls (Common string placeholders)
    hidden_null_patterns = ['?', 'n/a', 'nan', 'none', '', ' ', 'null']
    audit['hidden_nulls'] = df.apply(lambda x: x.astype(str).str.lower().strip().isin(hidden_null_patterns).sum())
    
    return audit.sort_values('null_pct', ascending=False)

# usage: audit_results = diagnostic_audit(df)
# display(audit_results)
```

### 2. The "Intermodular" Integration Check

This section ensures your df is ready for the modules that follow (e.g., your feature engineering or RAG project scripts).


```python
# Check for problematic column names (spaces, special chars) that break dot-notation
def check_column_health(df):
    """Checks for column names that may cause issues in code (e.g., spaces, special characters).
    Recommends renaming to snake_case for better modularity."""
    invalid_names = [col for col in df.columns if not re.match(r'^[a-z0-9_]+$', str(col).lower())]
    if invalid_names:
        print(f"⚠️ Recommendation: Rename {len(invalid_names)} columns to snake_case for modularity.")
        print(f"Sample invalid: {invalid_names[:5]}")

# Identify Potential Keys (High uniqueness, no nulls)
def suggest_indices(df):
    """Suggests potential primary keys based on uniqueness and null counts. Ideal keys have unique values and no nulls."""
    potential_keys = [col for col in df.columns if df[col].nunique() == len(df)]
    if potential_keys:
        print(f"🔑 Potential Primary Keys: {potential_keys}")

# check_column_health(df)
# suggest_indices(df)
```

### 3. Outlier and Distribution Scanning

Instead of just a histogram, we specifically look for skewness and outliers that will require scaling or clipping in your Scikit-Learn pipeline.


```python
def outlier_scan(df):
    """Detects columns with potential outliers using the IQR method."""
    numeric_df = df.select_dtypes(include=[np.number])
    outlier_report = {}
    
    for col in numeric_df.columns:
        Q1 = numeric_df[col].quantile(0.25)
        Q3 = numeric_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = numeric_df[(numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)]
        if not outliers.empty:
            outlier_report[col] = len(outliers)
            
    return pd.Series(outlier_report).sort_values(ascending=False)

# display(outlier_scan(df))
```

### 4. Categorical Consistency Check

Essential for your "Data Science Toolbox" to ensure your categorical encoding won't explode into too many dimensions.


```python
def categorical_health(df):
    """Checks categorical columns for high cardinality and hidden data quality issues (e.g., leading/trailing whitespace). Recommends cleaning or encoding strategies."""
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        n_uni = df[col].nunique()
        if n_uni > 50:
            print(f"❗ High Cardinality Alert: {col} has {n_uni} unique values.")
        if df[col].str.contains(r'^\s+|\s+$', regex=True).any():
            print(f"🧹 Cleanup Needed: {col} contains leading/trailing whitespace.")

# categorical_health(df)
```
