# REF: Data Types & Formatting
*Cookbook: `02_data_cleaning/`*

Type correction, text standardization, and column renaming.

📋 See `__REF_DATA_CLEANING_MASTER.ipynb` Steps 4 & 6 for decision guidance.


```python
import pandas as pd, numpy as np, re

# ─── 1. Logical Column Standardization ───────────────────────────────────────
def standardize_columns(df):
    """Enforces snake_case and removes special characters for modularity."""
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(' ', '_')
                  .str.replace(r'[^a-z0-9_]', '', regex=True))
    return df

# ─── 2. Robust Numeric Cleaning ──────────────────────────────────────────────
def clean_currency_col(series):
    """Removes currency symbols/commas and converts to nullable Int64 or float."""
    clean_series = series.astype(str).str.replace(r'[$,£€%\s,]', '', regex=True)
    return pd.to_numeric(clean_series, errors='coerce')

# ─── 3. Text Normalization ───────────────────────────────────────────────────
def normalize_strings(df):
    """Trims whitespace and lowercases all object columns."""
    str_cols = df.select_dtypes('object').columns
    df[str_cols] = df[str_cols].apply(lambda x: x.str.strip().str.lower())
    return df

# ─── 4. The 'Logic' Audit ────────────────────────────────────────────────────
def logic_audit(df):
    """Identify columns that are 'objects' but might be dates or numbers."""
    audit = pd.DataFrame({
        'current_dtype': df.dtypes,
        'n_unique':      df.nunique(),
        'sample_val':    df.iloc[0] if len(df) > 0 else None
    })
    return audit

# --- APPLICATION EXAMPLES ---
# df = standardize_columns(df)
# df['price'] = clean_currency_col(df['price'])
# df['join_date'] = pd.to_datetime(df['join_date'], errors='coerce')
```
