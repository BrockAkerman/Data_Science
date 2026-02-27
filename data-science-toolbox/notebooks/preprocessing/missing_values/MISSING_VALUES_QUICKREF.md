# Missing Value Handling Pipeline: Quick Reference

## Overview

This pipeline handles missing values with three feature-specific strategies and comprehensive audit logging. It replaces your ad-hoc, hardcoded missing value code with a production-ready system.

---

## Three Strategies Available

### Strategy 1: IMPUTE
Fill missing values using one of several methods, optionally with a binary missing indicator.

**Methods:**
- `median` - Fill with column median (robust to outliers)
- `mean` - Fill with column mean (assumes MCAR)
- `mode` - Fill with most frequent value (categorical data)
- `forward_fill` - Fill with previous value (time series)
- `custom` - Fill with arbitrary custom value

**Example:**
```python
'employment_duration': {
    'strategy': 'impute',
    'method': 'median',                # Fill with median
    'create_indicator': True,          # Track that it was missing
    'indicator_name': 'employment_duration_missing',
    'threshold_pct': 25.0,             # Warn if >25% missing
    'description': 'Likely predictive; missing ~2.8%; median-impute + indicator'
}
```

### Strategy 2: DROP
Remove rows with missing values in this feature.

**Use when:** Missing values are rare and rows are inconsequential to remove.

**Example:**
```python
'loan_amnt': {
    'strategy': 'drop',                # Remove rows with missing values
    'description': 'Retain feature; 1 missing value inconsequential; drop rows'
}
```

### Strategy 3: INDICATOR ONLY
Create a binary missing indicator WITHOUT imputing the values.

**Use when:** Missingness itself is informative (extreme missingness patterns).

**Example:**
```python
'historical_default': {
    'strategy': 'indicator_only',      # No imputation; create indicator only
    'create_indicator': True,
    'indicator_name': 'historical_default_missing',
    'threshold_pct': 100.0,            # Acknowledge 63.64% is extreme
    'description': 'Extreme missingness (~63.64%); indicator only, no imputation'
}
```

---

## Configuration Structure

```python
MISSING_VALUE_CONFIG = {
    'features': {
        'feature_name': {
            # Strategy and imputation method
            'strategy': 'impute',                    # 'impute', 'drop', 'indicator_only'
            'method': 'median',                      # 'median', 'mean', 'mode', 'forward_fill', 'custom'
            'custom_value': None,                    # If method='custom', set value here
            
            # Missing indicator
            'create_indicator': True,                # Create binary missing column
            'indicator_name': 'feature_missing',     # Name of indicator column
            
            # Quality control
            'threshold_pct': 25.0,                   # Warn if missing % exceeds this
            'description': 'Why this strategy for this feature'
        }
    },
    
    # Global settings
    'analysis_before': True,                         # Analyze missingness before imputing
    'fail_if_analysis_issues': False,                # Warn but don't fail on issues
}
```

---

## Basic Usage

### Option 1: Direct Function Call

```python
from missing_values_handling import analyze_missingness, handle_missing_values

# Step 1: Analyze missingness BEFORE making changes
analysis = analyze_missingness(data, MISSING_VALUE_CONFIG)

# Step 2: Handle missing values
data_clean, audit = handle_missing_values(data, MISSING_VALUE_CONFIG, analysis)

# Step 3: Check results
print(f"Rows removed: {audit['rows_removed']}")
print(f"Indicators created: {audit['indicators_created']}")
```

### Option 2: In a Full Pipeline

```python
from missing_values_handling import analyze_missingness, handle_missing_values

class DataCleaningPipeline:
    def execute(self, df):
        # Step 1: Analyze
        analysis = analyze_missingness(df, self.config['missing_value_handling'])
        
        # Step 2: Handle
        df_clean, audit = handle_missing_values(
            df,
            self.config['missing_value_handling'],
            analysis=analysis
        )
        
        self.audit_trail['missing_values'] = audit
        return df_clean, self.audit_trail
```

---

## Mapping Your Original Code to Configuration

### Original Code 1: employment_duration
```python
# BEFORE (hardcoded)
data['employment_duration_missing'] = data['employment_duration'].isna().astype(int)
data['employment_duration'] = data['employment_duration'].fillna(
    data['employment_duration'].median()
)
```

**BECOMES:**
```python
'employment_duration': {
    'strategy': 'impute',
    'method': 'median',
    'create_indicator': True,
    'indicator_name': 'employment_duration_missing',
    'threshold_pct': 25.0,
    'description': 'Likely predictive; missing ~2.8%; median-impute + indicator'
}
```

---

### Original Code 2: loan_int_rate
```python
# BEFORE (hardcoded)
data['loan_int_rate_missing'] = data['loan_int_rate'].isna().astype(int)
data['loan_int_rate'] = data['loan_int_rate'].fillna(data['loan_int_rate'].mean())
```

**BECOMES:**
```python
'loan_int_rate': {
    'strategy': 'impute',
    'method': 'mean',
    'create_indicator': True,
    'indicator_name': 'loan_int_rate_missing',
    'threshold_pct': 25.0,
    'description': 'Likely predictive; missing ~9.56%; mean-impute + indicator'
}
```

---

### Original Code 3: loan_amnt
```python
# BEFORE (hardcoded)
data = data.dropna(subset=['loan_amnt'])
```

**BECOMES:**
```python
'loan_amnt': {
    'strategy': 'drop',
    'method': None,
    'create_indicator': False,
    'threshold_pct': 5.0,
    'description': 'Retain feature; 1 missing value inconsequential; drop rows'
}
```

---

### Original Code 4: historical_default
```python
# BEFORE (hardcoded)
data['historical_default_missing'] = (data['historical_default'].isna().astype(int))
```

**BECOMES:**
```python
'historical_default': {
    'strategy': 'indicator_only',
    'create_indicator': True,
    'indicator_name': 'historical_default_missing',
    'threshold_pct': 100.0,
    'description': 'Extreme missingness (~63.64%); indicator only, no imputation'
}
```

---

### Original Code 5: Current_loan_status
```python
# BEFORE (hardcoded)
data = data.dropna(subset=['Current_loan_status'])
```

**BECOMES:**
```python
'Current_loan_status': {
    'strategy': 'drop',
    'method': None,
    'create_indicator': False,
    'threshold_pct': 5.0,
    'description': 'Retain feature; 4 missing values inconsequential; drop rows'
}
```

---

## Audit Trail Output

The function returns comprehensive audit information:

```python
audit = {
    'status': 'SUCCESS',                           # Overall status
    'rows_input': 10000,                           # Input rows
    'rows_output': 9999,                           # Output rows (after drops)
    'rows_removed': 1,                             # Rows removed by drop strategies
    
    'columns_input': 10,                           # Input columns
    'columns_output': 12,                          # Output columns (including indicators)
    
    'features_processed': [                        # All features handled
        'employment_duration',
        'loan_int_rate',
        'loan_amnt',
        'historical_default',
        'Current_loan_status'
    ],
    
    'indicators_created': [                        # New missing indicator columns
        'employment_duration_missing',
        'loan_int_rate_missing',
        'historical_default_missing'
    ],
    
    'details': {                                   # Per-feature breakdown
        'employment_duration': {
            'strategy': 'impute',
            'missing_before': 280,                 # ~2.8% of 10000
            'missing_after': 0,                    # All imputed
            'imputation_method': 'median',
            'indicator_created': True,
            'indicator_name': 'employment_duration_missing',
            'indicator_count': 280                 # 280 rows marked as originally missing
        },
        'loan_int_rate': {
            'strategy': 'impute',
            'missing_before': 956,                 # ~9.56%
            'missing_after': 0,
            'imputation_method': 'mean',
            'indicator_created': True,
            'indicator_name': 'loan_int_rate_missing',
            'indicator_count': 956
        },
        'loan_amnt': {
            'strategy': 'drop',
            'missing_before': 1,
            'missing_after': 0,
            'rows_removed': 1                      # 1 row dropped
        },
        # ... etc
    },
    
    'warnings': [],                                # Any threshold violations
    'errors': []                                   # Any errors encountered
}
```

---

## Analysis Output

Before handling, the pipeline analyzes missingness:

```python
analysis = {
    'total_rows': 10000,
    'total_columns': 10,
    
    'features_with_missing': {
        'employment_duration': {'count': 280, 'percent': 2.8, 'dtype': 'float64'},
        'loan_int_rate': {'count': 956, 'percent': 9.56, 'dtype': 'float64'},
        'loan_amnt': {'count': 1, 'percent': 0.01, 'dtype': 'float64'},
        'historical_default': {'count': 6364, 'percent': 63.64, 'dtype': 'float64'},
        'Current_loan_status': {'count': 4, 'percent': 0.04, 'dtype': 'float64'}
    },
    
    'features_with_no_missing': [
        'age', 'income', 'credit_score', 'loan_approved'
    ],
    
    'total_missing_cells': 8605,
    'warnings': []
}
```

---

## Logging Output Example

When you run the pipeline, it logs detailed information:

```
======================================================================
MISSINGNESS ANALYSIS (BEFORE IMPUTATION)
======================================================================

Dataset shape: 10000 rows × 10 columns

Columns with missing values: 5
Columns with no missing values: 5
Total missing cells: 8605

Detailed breakdown:
  historical_default       :  6364 missing (63.64%) [float64]
  loan_int_rate            :   956 missing ( 9.56%) [float64]
  employment_duration      :   280 missing ( 2.80%) [float64]
  Current_loan_status      :     4 missing ( 0.04%) [float64]
  loan_amnt                :     1 missing ( 0.01%) [float64]

======================================================================
MISSING VALUE HANDLING EXECUTION
======================================================================

employment_duration        : No missing values

──────────────────────────────────────────────────────────────────────
Feature: employment_duration
  Missing before: 280 (2.80%)
  Strategy: impute
  Method: Median imputation (value: 15.42)
  Indicator: Created 'employment_duration_missing' (280 = 1)
  Result: ✓ No missing values remain

──────────────────────────────────────────────────────────────────────
Feature: loan_int_rate
  Missing before: 956 (9.56%)
  Strategy: impute
  Method: Mean imputation (value: 10.84)
  Indicator: Created 'loan_int_rate_missing' (956 = 1)
  Result: ✓ No missing values remain

... [continues for each feature]

======================================================================
MISSING VALUE HANDLING SUMMARY
======================================================================
Status: SUCCESS
Features processed: 5
Indicators created: 3
  → employment_duration_missing, loan_int_rate_missing, historical_default_missing
Rows: 10000 → 9999 (removed: 1)
Columns: 10 → 13 (added: 3)
Warnings: 0
======================================================================
```

---

## Key Differences from Your Original Code

| Aspect | Original | Pipeline |
|--------|----------|----------|
| **Code lines** | ~15 hardcoded lines | 1-2 config lines per feature |
| **Reusability** | One-off script | Reusable function |
| **Auditability** | Manual tracking | Automatic audit trail |
| **Analysis** | Manual inspection | Automated analysis before handling |
| **Consistency** | Manual per-feature | Configuration-driven |
| **Error handling** | Basic | Comprehensive validation |
| **Documentation** | Inline comments | Docstrings + config descriptions |
| **Scalability** | Doesn't scale to many features | Scales to any number of features |

---

## Why This Matters for Recruiters

✅ **Data engineering best practices:**
- Configuration-driven code
- Separation of analysis and action
- Comprehensive audit trails

✅ **Machine learning rigor:**
- Missingness tracked with indicators
- MCAR vs MNAR distinction
- Pre-processing decisions documented

✅ **Production readiness:**
- Logging for debugging
- Error handling and validation
- Metrics for monitoring

✅ **Professional communication:**
- Clear function docstrings
- Configuration as documentation
- Structured output

---

## Files Provided

1. **missing_values_pipeline.ipynb** - Full Jupyter notebook with examples
2. **missing_values_handling.py** - Reusable Python module (importable)
3. **MISSING_VALUES_QUICKREF.md** - This quick reference guide

---

## Integration Checklist

- [ ] Copy config into your notebook
- [ ] Copy `analyze_missingness()` function
- [ ] Copy `handle_missing_values()` function
- [ ] Update to use your actual data and feature names
- [ ] Run analysis first to validate strategy decisions
- [ ] Run handling to execute strategies
- [ ] Check audit trail for results
- [ ] Drop unnecessary columns (e.g., customer_id after duplicate checking)
- [ ] Move to next pipeline step (outlier handling)
