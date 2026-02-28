# Section 3.4: Data Type Corrections - Integration Guide

## Position in Pipeline

```
Section 3.1: Duplicates (→ drop customer_id)
    ↓
Section 3.2: Missing Values (→ impute/drop/flag)
    ↓
[NEW] Section 3.4: Data Type Corrections ← YOU ARE HERE
    ↓
Section 3.3: Outliers (→ remove/cap/transform)
    ↓
Section 3.5: Value Corrections (→ standardize formats)
    ↓
Section 3.6: Validation & Summary
```

---

## Your Current Data Types

After Sections 3.1–3.2, your data looks like:

```python
customer_age              int64
customer_income           int64
home_ownership             str  ← Needs conversion to category
employment_duration     float64
loan_intent                str  ← Needs conversion to category
loan_grade                 str  ← Needs conversion to category
loan_amnt               float64
loan_int_rate           float64
term_years               int64
historical_default         str  ← Needs conversion to category
cred_hist_length         int64
Current_loan_status        str  ← Needs conversion to category
```

**What Section 3.4 does:**

1. **Converts string columns to category** (memory-efficient, better for modeling)
2. **Ensures numeric consistency** (int64 vs float64)
3. **Validates value ranges** (catches remaining data quality issues)
4. **Drops unnecessary columns** (if any remain)

---

## Three-Step Execution

### Step 1: Analyze

```python
# See current dtypes and what will change
type_analysis = analyze_data_types(data, DATA_TYPE_CONFIG)
```

**Output:**
```
Current dtypes:
  customer_age           : int64
  customer_income        : int64
  home_ownership         : object          ← Will convert to category
  employment_duration    : float64
  loan_intent            : object          ← Will convert to category
  ...
```

### Step 2: Correct

```python
# Apply all dtype corrections
data_typed, type_audit = correct_data_types(data, DATA_TYPE_CONFIG, type_analysis)
```

**Output:**
```
DATA TYPE CORRECTION SUMMARY
Status: SUCCESS
Dtype conversions: 5
  ✓ home_ownership: object → category (3 categories)
  ✓ loan_intent: object → category (6 categories)
  ✓ loan_grade: object → category (4 categories)
  ✓ historical_default: object → category (2 categories)
  ✓ Current_loan_status: object → category (3 categories)

Validation warnings: 0
Validation errors: 0
Rows: 32000 → 32000 (unchanged)
Columns: 13 → 13 (unchanged)
```

### Step 3: Continue Pipeline

```python
# Use data_typed in next section
outlier_analysis = analyze_outliers_all(data_typed, OUTLIER_CONFIG)
data_clean, outlier_audit = handle_outliers(data_typed, OUTLIER_CONFIG, outlier_analysis)
```

---

## Configuration Explained

### Numeric Columns

```python
'numeric_columns': {
    'customer_age': 'int64',         # No decimals → int64
    'employment_duration': 'float64', # May have decimals → float64
    'loan_amnt': 'float64',
    'loan_int_rate': 'float64',
    'term_years': 'int64',
    'cred_hist_length': 'int64',
}
```

**Why the distinction?**
- `int64`: Ages, years, counts (no decimals)
- `float64`: Percentages, durations with decimals

### Categorical Columns

```python
'categorical_columns': {
    'home_ownership': 'category',      # RENT, OWN, MORTGAGE
    'loan_intent': 'category',         # PERSONAL, EDUCATION, etc.
    'loan_grade': 'category',          # A, B, C, D (ordinal)
    'historical_default': 'category',  # Yes/No
    'Current_loan_status': 'category', # Active, Paid Off, Defaulted
}
```

**Why category dtype?**
- Memory efficient (categorical codes vs. repeated strings)
- Better for modeling (sklearn, xgboost understand categories)
- Prevents accidental arithmetic on categorical data
- Preserves order for ordinal variables (loan_grade)

### Validation Ranges

```python
'validate_ranges': {
    'customer_age': {'min': 18, 'max': 100},           # From outlier handling
    'customer_income': {'min': 0, 'max': None},        # Non-negative only
    'loan_amnt': {'min': 0, 'max': 900000},            # From outlier handling
    'loan_int_rate': {'min': 0, 'max': 100},           # 0-100%
    'term_years': {'min': 1, 'max': 50},               # Reasonable range
    'cred_hist_length': {'min': 0, 'max': 80},         # 0-80 years
}
```

Validation **warns** if values fall outside ranges (doesn't remove them—that's outlier handling's job).

---

## What Section 3.4 Does NOT Do

✓ Converts types  
✓ Validates ranges  
✓ Drops unnecessary columns  

✗ Remove outliers (that's Section 3.3)  
✗ Fix impossible values (that's Section 3.5)  
✗ Standardize formats like "RENT" vs "rent" (that's Section 3.5)  

---

## Output

The corrected dataframe has:

```python
# BEFORE Section 3.4
customer_age              int64
home_ownership             str    ← Takes more memory
loan_intent                str    ← Repeated strings
...

# AFTER Section 3.4  
customer_age              int64
home_ownership         category   ← Efficient, ordered
loan_intent            category   ← Efficient, ordered
...
```

**Benefits:**
- ~10-50% memory savings (especially with many categorical columns)
- Faster model training (xgboost, sklearn work natively with categories)
- Type safety (prevents accidental arithmetic on categories)
- Better downstream handling (Section 3.5 can rely on known types)

---

## Files Provided

1. **section_3_4_data_type_corrections.ipynb** - Jupyter notebook (ready to use)
2. **section_3_4_data_type_corrections.py** - Standalone Python module
3. **This guide** - Integration instructions

---

## Quick Start

```python
# After Section 3.2 (missing values):
data = data  # From previous section

# Section 3.4: Type corrections
type_analysis = analyze_data_types(data, DATA_TYPE_CONFIG)
data_typed, type_audit = correct_data_types(data, DATA_TYPE_CONFIG, type_analysis)

# Before Section 3.3 (outliers):
outlier_analysis = analyze_outliers_all(data_typed, OUTLIER_CONFIG)
data_clean, outlier_audit = handle_outliers(data_typed, OUTLIER_CONFIG, outlier_analysis)
```

---

## Customization

If your categorical columns have different values, just update the config:

```python
'categorical_columns': {
    'your_category_col': 'category',
}
```

If you need to add validation ranges:

```python
'validate_ranges': {
    'your_numeric_col': {'min': 0, 'max': 1000},
}
```

---

## Next Steps

After Section 3.4, you'll move to:
- **Section 3.5**: Value Corrections (standardize case, fix formats)
- **Section 3.6**: Final Validation & Summary
