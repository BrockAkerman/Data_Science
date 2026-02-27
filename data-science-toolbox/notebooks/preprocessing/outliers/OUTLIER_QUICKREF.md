# Outlier Handling Pipeline: Quick Reference

## Overview

This pipeline detects and treats outliers using a combination of statistical methods (IQR, Z-score) and domain knowledge. Unlike simple rule-based approaches, it's **feature-specific and justifiable**.

---

## Five Treatment Strategies Available

### Strategy 1: REMOVE
Delete rows containing outliers. Best when outliers are implausible (data errors) or rare.

**Your examples:**
- `customer_age` < 18 (illegal) or > 100 (implausible) → Remove rows
- `employment_duration` > 100 years → Remove row
- `loan_amnt` at 1M, 3.5M (eCDF discontinuity) → Remove rows

```python
'customer_age': {
    'detection_method': 'domain_based',
    'treat_method': 'remove',
    'lower_bound': 18,                # Remove if < 18
    'upper_bound': 100,               # Remove if > 100
    'description': 'Age < 18 (legal threshold) or > 100 (implausible). Domain-based removal.'
}
```

### Strategy 2: CAP
Replace outliers with fence values (IQR bounds). Best when outliers are extreme but legitimate.

**Use case:** Age 0 replaced with Q1 boundary; income capped at upper fence.

```python
'income_capped': {
    'detection_method': 'iqr',
    'treat_method': 'cap',
    'iqr_multiplier': 1.5,            # Standard Tukey fences
    'description': 'Cap extreme values at IQR fences; retain information about direction.'
}
```

### Strategy 3: TRANSFORM
Apply mathematical transformation to reduce skewness. Best for right-skewed distributions.

**Your example:**
- `customer_income` right-skewed → Apply log transformation → Creates `customer_income_transformed`

```python
'customer_income': {
    'detection_method': 'none',       # No removal; transform instead
    'treat_method': 'transform',
    'transform_type': 'log1p',        # log(x+1); handles zeros
    'description': 'Right-skewed. Log transformation stabilizes variance.'
}
```

**Transform options:**
- `log1p`: Log(1 + x); handles zeros and negative values
- `sqrt`: Square root; gentler than log
- `yeo-johnson`: Power transformation to achieve normality (advanced)

### Strategy 4: RETAIN
Keep outliers unchanged. Best when outliers are legitimate or informative.

**Your examples:**
- `loan_int_rate` outliers (rates > 20%) → Real high-risk loans, retain
- `cred_hist_length` outliers → Natural variation (life expectancy), retain

```python
'loan_int_rate': {
    'detection_method': 'iqr',
    'treat_method': 'retain',         # Keep all values
    'description': 'Outliers represent legitimate high-risk loans. Retain.'
}
```

### Strategy 5: FLAG
Create binary outlier indicator without removing/transforming.

**Use case:** Track which values were originally flagged as outliers, for later investigation.

```python
'suspicious_feature': {
    'detection_method': 'iqr',
    'treat_method': 'flag',
    'create_indicator': True,
    'indicator_name': 'suspicious_feature_is_outlier'
}
```

---

## Detection Methods Explained

| Method | How It Works | When To Use | Pros | Cons |
|--------|---|---|---|---|
| **iqr** | Tukey fences: Q1 ± 1.5×IQR | Robust, distribution-free | Works for any shape | May flag normal extremes |
| **zscore** | Values with \|z\| > 3.0 | Assume normality | Simple | Sensitive to non-normal data |
| **domain_based** | Apply business rules (e.g., age < 18) | Know impossible ranges | Most accurate | Requires domain knowledge |
| **hybrid** | Combine methods | Complex scenarios | Flexible | Harder to interpret |
| **none** | Skip detection | Transform instead | Clean, simple | Can't identify problems |

### Your Feature-Specific Choices:

| Feature | Detection | Why |
|---------|-----------|-----|
| `customer_age` | **domain_based** | Legal/biological bounds known; IQR insufficient |
| `customer_income` | **none** | Right-skew is natural; no outliers—transform instead |
| `employment_duration` | **domain_based** | >100 years is implausible; remove |
| `loan_amnt` | **hybrid** | IQR + eCDF discontinuity analysis; complex |
| `loan_int_rate` | **iqr** | Rates > 20% pass IQR test; legitimate high-risk |
| `cred_hist_length` | **iqr** | Outliers are natural (life expectancy) |

---

## Configuration Structure

```python
OUTLIER_CONFIG = {
    'features': {
        'feature_name': {
            # Detection
            'detection_method': 'iqr',                   # iqr, zscore, domain_based, hybrid, none
            
            # Treatment
            'treat_method': 'remove',                     # remove, cap, transform, retain, flag
            
            # Domain bounds (for domain_based detection)
            'lower_bound': 18,                            # Minimum plausible value
            'upper_bound': 100,                           # Maximum plausible value
            
            # IQR parameters
            'iqr_multiplier': 1.5,                        # 1.5 = standard, 3.0 = extreme only
            
            # Transform parameters
            'transform_type': 'log1p',                    # log1p, sqrt, yeo-johnson
            
            # Feature creation
            'create_indicator': False,                    # Create binary outlier flag
            'indicator_name': 'feature_is_outlier',
            
            # Documentation
            'description': 'Why this strategy for this feature'
        }
    }
}
```

---

## Basic Usage

```python
from outlier_handling import analyze_outliers_all, handle_outliers

# Step 1: Analyze outliers BEFORE treating
analysis = analyze_outliers_all(data, OUTLIER_CONFIG)

# Step 2: Treat based on configuration
data_clean, audit = handle_outliers(data, OUTLIER_CONFIG, analysis)

# Step 3: Check results
print(f"Rows removed: {audit['rows_removed']}")
print(f"Features transformed: {audit['features_transformed']}")
```

---

## Audit Trail Output

```
======================================================================
OUTLIER ANALYSIS (BEFORE TREATMENT)
======================================================================

customer_age:
  Detection: domain_based
  Outliers (IQR): 9 (0.090%)
  Range: [5.00, 180.00]
  Q1=25.00, Q3=47.00
  Fences: [8.00, 64.00]

customer_income:
  Detection: none
  [skipped - no detection]

loan_amnt:
  Detection: hybrid
  Outliers (IQR): 5 (0.050%)
  Range: [1000.00, 3500000.00]
  Fences: [2500.00, 62500.00]

======================================================================
OUTLIER TREATMENT EXECUTION
======================================================================

Feature: customer_age
  Detection: domain_based
  Treatment: remove
  Lower bound: customer_age >= 18
  Upper bound: customer_age <= 100
  Rows removed: 9

Feature: customer_income
  Detection: none
  Treatment: transform
  Applied log1p transformation → 'customer_income_transformed'

Feature: loan_amnt
  Detection: hybrid
  Treatment: remove
  Upper bound: loan_amnt <= 900000
  Rows removed: 3

Feature: loan_int_rate
  Detection: iqr
  Treatment: retain
  Outliers retained as legitimate values

======================================================================
OUTLIER HANDLING SUMMARY
======================================================================
Features processed: 6
Features transformed: 1
  → customer_income_transformed
Rows: 10000 → 9988 (removed: 12)
======================================================================
```

---

## Your Feature-Specific Config (Ready to Use)

```python
OUTLIER_CONFIG = {
    'features': {
        'customer_age': {
            'detection_method': 'domain_based',
            'treat_method': 'remove',
            'lower_bound': 18,
            'upper_bound': 100,
            'description': 'Age < 18 (legal threshold) or > 100 (implausible). Domain-based removal.'
        },
        'customer_income': {
            'detection_method': 'none',
            'treat_method': 'transform',
            'transform_type': 'log1p',
            'description': 'Right-skewed. Log transformation stabilizes variance; retain all values.'
        },
        'employment_duration': {
            'detection_method': 'domain_based',
            'treat_method': 'remove',
            'upper_bound': 100,
            'description': 'Duration > 100 years is implausible data entry error. Remove.'
        },
        'loan_amnt': {
            'detection_method': 'hybrid',
            'treat_method': 'remove',
            'upper_bound': 900000,
            'description': 'Values 1M, 3.5M show eCDF discontinuity (different process). Remove.'
        },
        'loan_int_rate': {
            'detection_method': 'iqr',
            'treat_method': 'retain',
            'description': 'Outliers (>20%) are legitimate high-risk loans. Retain.'
        },
        'cred_hist_length': {
            'detection_method': 'iqr',
            'treat_method': 'retain',
            'description': 'Outliers reflect natural distribution (life expectancy). Retain.'
        }
    }
}
```

---

## IQR Multiplier Explained

**Tukey's Standard (multiplier=1.5)**
- Flags ~0.7% of normally distributed data as outliers
- Typical in exploratory analysis
- May flag legitimate extreme values

**Conservative (multiplier=3.0)**
- Flags only extreme outliers (~0.0% of normal data)
- Use when you want to preserve most data
- Focuses on severe anomalies only

```python
# Standard: flags ~1% as outliers
detect_outliers_iqr(series, multiplier=1.5)

# Conservative: flags only extreme outliers
detect_outliers_iqr(series, multiplier=3.0)
```

---

## Common Decisions

### "Should I remove or cap?"

| Situation | Decision | Why |
|-----------|----------|-----|
| Age 17, 121 | **Remove** | Outside legal/biological bounds |
| Income $10M | **Cap** or **Transform** | Extreme but possible; preserve direction |
| 0 employees (data error) | **Remove** | Clear error |
| Rare but real value | **Retain** | Legitimate information |

### "Should I transform or remove?"

| Scenario | Action | Why |
|----------|--------|-----|
| Right-skewed income, no errors | **Transform (log)** | Stabilize variance; retain all data |
| Income with clear $0 entries | **Remove rows** | Errors shouldn't be transformed |
| Age with typos (999 years) | **Remove rows** | Implausible; clean data better |

### "When to create missing indicators?"

Create indicators when missingness signal might be predictive:
- Employment duration missing → May indicate new worker
- Income missing → May indicate unemployment
- BUT: Rate outliers → No indicator needed (rate isn't "missing")

---

## Production Checklist

- [ ] Analyze outliers BEFORE treating (understand patterns first)
- [ ] Document domain knowledge (bounds, why removal is justified)
- [ ] Test on sample data to verify decisions
- [ ] Run full pipeline audit trail (reproducible)
- [ ] Validate post-treatment (shapes, ranges)
- [ ] Check for unintended consequences (new missing values, data loss)
- [ ] Report row counts before/after
- [ ] Archive configuration for reproducibility

---

## Files Provided

1. **outlier_handling_pipeline.ipynb** - Full Jupyter notebook with examples
2. **outlier_handling.py** - Reusable Python module (importable)
3. **OUTLIER_QUICKREF.md** - This quick reference guide
