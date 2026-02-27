# Data Type Conversion: Integration with Outlier Pipeline

## The Problem

Your columns like `customer_income`, `loan_amnt`, and `loan_int_rate` are stored as **strings**, not numbers:

```python
# What your data looks like
customer_income: ['€45000', '€62000', '€78000']  # String, not numeric
loan_amnt: ['€15000', '€25000', '€35000']        # String, not numeric
loan_int_rate: ['8.5%', '10.2%', '12.0%']        # String, not numeric
```

The outlier pipeline checks `pd.api.types.is_numeric_dtype()` and **skips non-numeric columns**, which is why you're seeing:

```
✗ customer_income: Not numeric; skipping
✗ loan_amnt: Not numeric; skipping
```

---

## The Solution

Convert string columns to numeric **BEFORE** running outlier analysis.

### Step 1: Add Conversion Functions to Your Notebook

```python
def currency_to_numeric(series: pd.Series) -> pd.Series:
    """Convert currency strings (€45000) to numeric."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    
    # Remove currency symbols and commas
    cleaned = series.astype(str).str.replace(r'[€$£¥\s]', '', regex=True)
    cleaned = cleaned.str.replace(',', '', regex=False)
    return pd.to_numeric(cleaned, errors='coerce')


def percentage_to_numeric(series: pd.Series) -> pd.Series:
    """Convert percentage strings (8.5%) to numeric."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    
    # Remove % sign
    cleaned = series.astype(str).str.replace(r'[%\s]', '', regex=True)
    return pd.to_numeric(cleaned, errors='coerce')


def prepare_data_for_outlier_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Convert string columns to numeric before outlier analysis."""
    df_clean = df.copy()
    
    # Customize these lists based on YOUR actual columns
    currency_columns = ['customer_income', 'loan_amnt']
    percentage_columns = ['loan_int_rate']
    
    # Convert currency columns
    for col in currency_columns:
        if col in df_clean.columns and not pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = currency_to_numeric(df_clean[col])
            logger.info(f"✓ Converted {col} to numeric")
    
    # Convert percentage columns
    for col in percentage_columns:
        if col in df_clean.columns and not pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = percentage_to_numeric(df_clean[col])
            logger.info(f"✓ Converted {col} to numeric")
    
    return df_clean
```

### Step 2: Use Conversion BEFORE Outlier Analysis

```python
# BEFORE (doesn't work - columns are strings)
outlier_analysis = analyze_outliers_all(data, OUTLIER_CONFIG)

# AFTER (works - convert first)
data_prepared = prepare_data_for_outlier_analysis(data)
outlier_analysis = analyze_outliers_all(data_prepared, OUTLIER_CONFIG)
```

### Step 3: Use Prepared Data Throughout

```python
# Analyze
outlier_analysis = analyze_outliers_all(data_prepared, OUTLIER_CONFIG)

# Treat
data_clean, audit = handle_outliers(data_prepared, OUTLIER_CONFIG, outlier_analysis)
```

---

## Complete Workflow

```python
import pandas as pd
from outlier_handling import analyze_outliers_all, handle_outliers

# 1. Load raw data
data = pd.read_csv('loan_data.csv')
print("Raw data types:")
print(data.dtypes)

# 2. Convert string columns to numeric
data_prepared = prepare_data_for_outlier_analysis(data)
print("\nAfter conversion:")
print(data_prepared.dtypes)

# 3. Analyze outliers
outlier_analysis = analyze_outliers_all(data_prepared, OUTLIER_CONFIG)

# 4. Treat outliers
data_clean, audit = handle_outliers(data_prepared, OUTLIER_CONFIG, outlier_analysis)

# 5. Check results
print(f"\nRows removed: {audit['rows_removed']}")
print(f"Features transformed: {audit['features_transformed']}")
```

---

## Customization

### For Different Column Names

If your columns are named differently, update `prepare_data_for_outlier_analysis()`:

```python
def prepare_data_for_outlier_analysis(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    
    # YOUR actual column names
    currency_columns = [
        'monthly_income',      # Instead of customer_income
        'requested_amount',    # Instead of loan_amnt
    ]
    
    percentage_columns = [
        'interest_rate',       # Instead of loan_int_rate
        'apr',
    ]
    
    for col in currency_columns:
        if col in df_clean.columns and not pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = currency_to_numeric(df_clean[col])
    
    for col in percentage_columns:
        if col in df_clean.columns and not pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = percentage_to_numeric(df_clean[col])
    
    return df_clean
```

### For Additional Formats

If you have other string formats, add custom converters:

```python
def custom_to_numeric(series: pd.Series, pattern: str) -> pd.Series:
    """Convert custom string format to numeric."""
    cleaned = series.astype(str).str.replace(pattern, '', regex=True)
    return pd.to_numeric(cleaned, errors='coerce')

# Example: Handle "1,234.56 USD"
data_prepared['amount_usd'] = custom_to_numeric(
    data['amount_str'],
    r'[\s,USD]'  # Remove spaces, commas, "USD"
)
```

---

## What Gets Converted

### Currency Conversion Example
```python
# Input
'€45000', '€62,000.50', '$100,000'

# Output (numeric)
45000.0, 62000.5, 100000.0
```

### Percentage Conversion Example
```python
# Input
'8.5%', '10.2%', '12.0%'

# Output (numeric)
8.5, 10.2, 12.0
```

---

## Error Handling

The conversion functions use `errors='coerce'`, which means:
- Invalid strings → NaN (not errors)
- Empty strings → NaN
- Already numeric → Passed through unchanged

Example:
```python
s = pd.Series(['€50000', 'invalid', '€75000'])
result = currency_to_numeric(s)

# Result:
# 0    50000.0
# 1       NaN    ← Invalid string becomes NaN
# 2    75000.0
```

These NaNs will be handled in your missing value section (Section 3.2).

---

## Files Provided

1. **data_type_conversions.py** - Standalone conversion functions
2. **outlier_handling_pipeline.ipynb** - Updated notebook with integrated conversions
3. **This guide** - Integration instructions

---

## Quick Start (TL;DR)

```python
# 1. Convert string columns
data = prepare_data_for_outlier_analysis(data)

# 2. Analyze outliers (now columns are numeric!)
analysis = analyze_outliers_all(data, OUTLIER_CONFIG)

# 3. Treat outliers
data_clean, audit = handle_outliers(data, OUTLIER_CONFIG, analysis)

# Done! ✓
```

---

## Next Steps

After outlier handling, you'll move to **Section 3.4: Data Type Corrections**, where you can formally standardize all dtypes (integers, floats, datetimes, etc.) as part of the overall pipeline.
