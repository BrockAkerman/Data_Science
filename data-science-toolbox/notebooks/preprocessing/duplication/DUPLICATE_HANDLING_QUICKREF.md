# Duplicate Handling Pipeline: Quick Reference

## Overview

This pipeline provides production-ready duplicate detection and removal with comprehensive audit logging. It's designed to demonstrate professional data engineering practices to recruiters.

---

## What It Does

### Check 1: **Exact Row Duplicates**
- Identifies rows that are identical across all columns
- Logs sample of duplicates for inspection
- Reports count and indices

### Check 2: **Customer-Level Duplicates**
- Identifies customers with multiple records in the dataset
- Distinguishes between legitimate multiple loans vs. data duplication
- Reports distribution of duplicates per customer

### Removal & Validation**
- Removes duplicates using pandas `drop_duplicates()`
- Validates no duplicates remain after cleaning
- Logs all metrics for audit trail

---

## Configuration

```python
DUPLICATE_CONFIG = {
    'check_exact_rows': True,              # Check exact row duplicates
    'check_customer_level': True,          # Check customer-level duplicates
    'customer_id_column': 'customer_id',   # Name of customer ID column
    'keep': 'first',                       # Keep 'first' or 'last' occurrence
    'fail_if_duplicates_remain': True,     # Raise error if cleanup fails
    'subset': None,                        # None = all columns; or list of column names
}
```

---

## Basic Usage

### Option 1: Use the Function Directly

```python
from duplicate_handling import handle_duplicates

# Load your data
data = pd.read_csv('loan_data.csv')

# Configure
config = {
    'check_exact_rows': True,
    'check_customer_level': True,
    'customer_id_column': 'customer_id',
    'keep': 'first',
    'fail_if_duplicates_remain': True,
}

# Run
data_clean, audit = handle_duplicates(data, config)

# Check results
print(f"Rows removed: {audit['rows_removed']}")
print(f"Status: {audit['status']}")
```

### Option 2: Use the Full Pipeline Class

```python
from duplicate_handling import DataCleaningPipeline

# Configure
config = {
    'duplicate_handling': DUPLICATE_CONFIG,
    # Future: 'missing_value_handling': {...},
    # Future: 'outlier_handling': {...},
}

# Initialize
pipeline = DataCleaningPipeline(config)

# Execute
data_clean, audit_trail = pipeline.execute(data)

# Get summary
print(pipeline.get_summary())
```

---

## What The Audit Trail Contains

```python
audit = {
    'status': 'SUCCESS',                          # Overall status
    'rows_input': 10000,                          # Input row count
    'rows_output': 9994,                          # Output row count
    'rows_removed': 6,                            # Rows removed
    'checks_performed': [                         # Which checks ran
        'exact_row_check', 
        'customer_level_check'
    ],
    'duplicate_details': {                        # Detailed results
        'exact_duplicates': {
            'count': 6,
            'duplicate_pairs': 3,
            'sample_indices': [...]
        },
        'customer_level': {
            'duplicate_customers': 0,
            'duplicate_rows': 0,
            'distribution': {}
        }
    },
    'errors': []                                  # Any errors encountered
}
```

---

## Example: With Your Data

```python
# Load your loan data
data = pd.read_csv('your_loan_data.csv')

# Define config (matches your requirements)
config = {
    'check_exact_rows': True,           # Find exact row matches
    'check_customer_level': True,       # Find multiple records per customer
    'customer_id_column': 'customer_id',
    'keep': 'first',                    # Keep first occurrence
    'fail_if_duplicates_remain': True,  # Fail if we don't clean properly
}

# Run
data_clean, audit = handle_duplicates(data, config)

# The audit shows:
# - 6 customer IDs appeared multiple times
# - Each duplicate pair had identical values across all features
# - After removal: X rows → Y rows
```

---

## Logging Output Example

```
✓ Exact row check: Found 6 duplicate rows
  Sample of duplicates (first 6 rows):
    customer_id  age  income  credit_score  loan_amount  approved
  0        1001   28   45000            720        15000         1
  1        1001   28   45000            720        15000         1
  2        1003   42   78000            750        35000         1
  3        1003   42   78000            750        35000         1
  ...

✓ Customer-level check: 6 customers appear multiple times
  Total rows involved in customer-level duplicates: 6
  Distribution: {1001: 2, 1003: 3, ...}

✓ Duplicate removal: 6 rows removed
  Input rows: 10000 → Output rows: 9994

✓ VALIDATION PASSED: No duplicates remain

======================================================================
DUPLICATE HANDLING SUMMARY
======================================================================
Status: SUCCESS
Rows removed: 6
Input shape: (10000, 12)
Output shape: (9994, 12)
======================================================================
```

---

## Integration with Full Pipeline

This function is designed to be one step in a larger pipeline:

```python
class DataCleaningPipeline:
    def execute(self, df):
        df_current = df.copy()
        
        # Step 1: Handle duplicates (THIS FUNCTION)
        df_current, audit = handle_duplicates(df_current, ...)
        self.audit_trail['duplicates'] = audit
        
        # Step 2: Handle missing values (FUTURE)
        # df_current, audit = handle_missing_values(df_current, ...)
        # self.audit_trail['missing_values'] = audit
        
        # Step 3: Handle outliers (FUTURE)
        # df_current, audit = handle_outliers(df_current, ...)
        # self.audit_trail['outliers'] = audit
        
        return df_current, self.audit_trail
```

---

## Why This Matters for Recruiters

✓ **Reproducible:** Configuration-driven, not hardcoded
✓ **Auditable:** Every step logged with detailed metrics
✓ **Testable:** Returns both data and audit trail for validation
✓ **Scalable:** Works with large datasets
✓ **Professional:** Production-ready error handling
✓ **Documented:** Docstrings and comments throughout
✓ **Modular:** Integrates into full pipeline cleanly

---

## Common Customizations

### Check only certain columns for duplicates:
```python
config['subset'] = ['customer_id', 'age', 'income']
```

### Keep last occurrence instead of first:
```python
config['keep'] = 'last'
```

### Don't fail if duplicates remain (warning only):
```python
config['fail_if_duplicates_remain'] = False
```

### Skip customer-level check:
```python
config['check_customer_level'] = False
```

---

## Files Provided

1. **duplicate_handling_pipeline.ipynb** - Full Jupyter notebook with examples
2. **duplicate_handling.py** - Reusable Python module (importable)
3. **QUICKREF.md** - This quick reference guide

---

## Next Steps in Pipeline

After duplicate handling:
1. ✓ **Duplicate Handling** (this)
2. → **Missing Value Strategy**
3. → **Outlier Handling**
4. → **Data Type Corrections**
5. → **Value Corrections**
6. → **Feature Engineering**
