# REF: Outlier Detection & Treatment
*Cookbook: `02_data_cleaning/`*

Ad-hoc outlier snippets. Grab the method that fits your column.

📋 See `__REF_DATA_CLEANING_MASTER.ipynb` Step 5 for the decision guide.  
📋 See `TMPL_PIPELINE_outlier_handling.ipynb` for the audit-logged pipeline version.

### 1. The Unified Outlier Audit
Instead of running separate scripts for every column, this function provides a high-level "Heat Map" of where your data is leaking variance.


```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest

def outlier_audit(df):
    """Provides a summary of outliers across all numeric columns using IQR."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    report = []
    
    for col in numeric_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - (1.5 * iqr), q3 + (1.5 * iqr)
        
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        report.append({
            'column': col,
            'outlier_count': len(outliers),
            'pct_outliers': round((len(outliers) / len(df)) * 100, 2),
            'min': df[col].min(),
            'max': df[col].max()
        })
        
    return pd.DataFrame(report).sort_values('outlier_count', ascending=False)

# Usage: display(outlier_audit(df))
```

### 2. Treatment: Winsorization (Capping)
Deleting rows is often too aggressive for technical business projects. Winsorization "caps" the outliers at a specific percentile (e.g., the 1st and 99th), preserving the data while removing the extreme skew.


```python
def treat_outliers_winsorize(df, col, limits=(0.01, 0.01)):
    """Caps extreme values at the specified percentiles."""
    df[f'{col}_clean'] = stats.mstats.winsorize(df[col], limits=limits)
    print(f"✅ {col} capped at {limits} percentiles.")
    return df

# Usage: df = treat_outliers_winsorize(df, 'revenue', limits=(0.05, 0.05))
```

### 3. Multivariate Detection (Isolation Forest)

For complex datasets where an outlier isn't just a "big number" but a "weird combination of numbers," we use an Isolation Forest.


```python
def detect_multivariate_outliers(df, contamination=0.05):
    """Uses an Isolation Forest to flag rows that are statistically 'lonely'."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    iso = IsolationForest(contamination=contamination, random_state=42)
    
    # -1 is outlier, 1 is inlier
    df['is_outlier'] = iso.fit_predict(df[num_cols].fillna(0))
    
    num_found = (df['is_outlier'] == -1).sum()
    print(f"🌲 Isolation Forest flagged {num_found} rows as multivariate outliers.")
    return df
```

| Situation                                  | Strategy                                      | Modular Tool                         |
|--------------------------------------------|-----------------------------------------------|--------------------------------------|
| Impossible values (e.g., Age 999)          | Drop row (Measurement Error)                  | `df.drop()`                          |
| Extreme but possible ($10M Income)         | Winsorize (Cap extreme influence)             | `treat_outliers_winsorize()`         |
| Right-skewed, legitimate data             | Log Transform to normalize                    | `np.log1p()`                         |
| Complex / Multivariate outliers           | Flag and keep for model                       | `detect_multivariate_outliers()`     |
| Uncertain / Natural Variance              | Retain as-is                                  | (No action)                          |
