# __REF: Data Cleaning Master Reference
*The authoritative guide to the data cleaning process. Use this as your mental model before touching any data.*

---

## File Taxonomy in This Folder

| Prefix | Purpose | When to Open |
|---|---|---|
| `__REF_` | Process knowledge — ordered steps, checklists, decision guides | **Before** starting cleaning |
| `REF_` | Ad-hoc snippets — for loops, quick utilities, one-liners | **During** cleaning, grab what you need |
| `TMPL_PIPELINE_` | Production pipeline functions — config-driven, audit-logged | **When** your project needs reproducible, recruiter-ready code |

---

## The Data Cleaning Order

Steps must happen in this sequence. Each step depends on the one before it.

```
1. Inspect & Precheck          ← understand before touching
2. Remove Duplicates           ← clean the row space
3. Handle Missing Values       ← fill or flag gaps
4. Fix Data Types              ← ensures outlier detection works correctly
5. Detect & Treat Outliers     ← now that types are correct
6. Value Correction            ← fix inconsistencies, standardize text
7. Rename & Reorganize         ← clean column names, reorder
8. Validate & Export           ← confirm quality, save
```

**Why this order matters:**
- Type correction before outlier detection — IQR on a string column fails silently
- Missingness before types — imputing a string '99' as median breaks
- Duplicates first — don't waste imputation on rows you'll drop anyway

---

## Decision Framework: Which Tool Do I Reach For?

```
Simple project, quick iteration?     →  REF_ snippets
Need audit trail for stakeholders?   →  TMPL_PIPELINE_
Unsure what strategy to use?         →  __REF_DATA_CLEANING_MASTER (you are here)
```

---
## Step 1 — Inspect & Precheck

> *Look before you touch. Form hypotheses. Understand the data's shape, types, and pathologies.*

**Checklist:**
- [ ] Load and confirm shape, dtypes, non-null counts
- [ ] Check distributions of all numeric features
- [ ] Check value frequencies of categorical features
- [ ] Identify nulls per column (count + %)
- [ ] Identify duplicates (exact rows + entity level)
- [ ] Inspect target variable distribution
- [ ] Check time ordering if time-series data
- [ ] Note relationships between features (correlation heatmap)
- [ ] Document your initial observations — surprises, concerns, hypotheses

📋 **REF file:** `REF_data_precheck.ipynb`


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load and assert ───────────────────────────────────────────────────────
# df = pd.read_csv("")
# assert len(df) > 0, "Dataset failed to load or is empty"

# ── Structural overview ───────────────────────────────────────────────────
# df.info()                      # dtypes, non-null counts
# df.describe(include='all').T   # descriptive stats
# df.head()                      # first look at structure
# df.shape                       # dimensionality

# ── Missing values ────────────────────────────────────────────────────────
# df.isnull().sum()              # nulls per column
# df.isnull().mean() * 100       # null % per column

# ── Duplicates ────────────────────────────────────────────────────────────
# df.duplicated().sum()          # exact row dupe count

# ── Frequency distribution ────────────────────────────────────────────────
# df['column'].value_counts()    # categorical inspection

# ── Correlation heatmap ───────────────────────────────────────────────────
# corr = df.select_dtypes(include=np.number).corr()
# plt.figure(figsize=(10,8))
# sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
# plt.show()
```

---
## Step 2 — Remove Duplicates

> *Duplicates inflate counts, distort statistics, and leak into train/test splits.*

**Decision guide:**
- Are these exact copies (all columns match)? → Drop all but first
- Are they entity-level duplicates (same customer_id, different rows)? → Investigate before dropping
- Do duplicates look like separate legitimate events? → Keep, don't drop

**Checklist:**
- [ ] Count exact row duplicates
- [ ] Inspect a sample before removing
- [ ] Check entity-level duplicates if an ID column exists
- [ ] Confirm count after removal
- [ ] Document how many were removed and why

📋 **REF file:** `REF_duplicate_handling.ipynb`  
📋 **Pipeline file:** `TMPL_PIPELINE_duplicate_handling.ipynb`


```python
# ── Locate ────────────────────────────────────────────────────────────────
# df.duplicated().value_counts()
# df[df.duplicated(keep=False)].sort_values(list(df.columns))

# ── Check before removing ─────────────────────────────────────────────────
# print(f"Duplicates found: {df.duplicated().sum()}")

# ── Remove ────────────────────────────────────────────────────────────────
# df = df.drop_duplicates()                            # all columns
# df = df.drop_duplicates(subset=['id'], keep='first') # entity-level

# ── Confirm ───────────────────────────────────────────────────────────────
# print(f"Duplicates remaining: {df.duplicated().sum()}")
```

---
## Step 3 — Handle Missing Values

> *Missing data is never random by accident. Understand why before deciding what to do.*

**Decision guide (by missingness %):**

| % Missing | Recommended Strategy |
|---|---|
| < 5% | Median (numeric) or Mode (categorical) fill |
| 5–20% | Impute + create missingness indicator column |
| 20–40% | KNN or model-based imputation + indicator |
| > 40% | Drop column, or indicator-only (missingness IS the signal) |
| Time series | Forward fill (`ffill`), then backward fill |

**MCAR / MAR / MNAR:**
- MCAR (Missing Completely at Random) → mean/median imputation is safe
- MAR (Missing at Random, depends on other cols) → model-based imputation
- MNAR (Missing Not at Random, related to the value itself) → flag as its own feature

**Checklist:**
- [ ] Audit nulls per column (count + %)
- [ ] Decide strategy per column — document reasoning
- [ ] Create missingness indicator before imputing (when missingness is informative)
- [ ] Fit imputers on train only (never on val/test)
- [ ] Confirm no unexpected nulls remain

📋 **REF file:** `REF_handle_missing_values.ipynb`  
📋 **Pipeline file:** `TMPL_PIPELINE_missing_values.ipynb`


```python
# ── Null audit ────────────────────────────────────────────────────────────
# null_df = pd.DataFrame({
#     'null_count': df.isnull().sum(),
#     'null_pct':   (df.isnull().mean() * 100).round(2)
# }).query('null_count > 0').sort_values('null_pct', ascending=False)
# display(null_df)

# ── Strategy: median/mode fill ────────────────────────────────────────────
# df['num_col'] = df['num_col'].fillna(df['num_col'].median())
# df['cat_col'] = df['cat_col'].fillna(df['cat_col'].mode()[0])

# ── Strategy: flag + fill ─────────────────────────────────────────────────
# df['col_was_null'] = df['col'].isnull().astype(int)
# df['col'] = df['col'].fillna(df['col'].median())

# ── Strategy: drop rows ───────────────────────────────────────────────────
# df = df.dropna(subset=['critical_col'])

# ── Strategy: ffill (time-series) ─────────────────────────────────────────
# df['col'] = df['col'].fillna(method='ffill')

# ── Strategy: KNN imputation ──────────────────────────────────────────────
# from sklearn.impute import KNNImputer
# knn = KNNImputer(n_neighbors=5)
# df[num_cols] = knn.fit_transform(df[num_cols])   # fit on train only!
```

---
## Step 4 — Fix Data Types

> *Fix types before outlier detection. IQR on a numeric-looking string column fails silently.*

**Common issues:**
- Numeric columns stored as `object` (e.g., `'$12,000'`, `'99.5%'`)
- Date columns stored as string
- IDs stored as `int` when they should be `str`
- Low-cardinality strings that should be `category`
- Inconsistent capitalization / whitespace in string columns

**Checklist:**
- [ ] Review `df.dtypes` for surprises
- [ ] Strip currency symbols, commas, % before `to_numeric()`
- [ ] Convert date columns with `pd.to_datetime(errors='coerce')`
- [ ] Cast low-cardinality strings to `category` (saves memory)
- [ ] Strip whitespace + normalize case on all string columns
- [ ] Rename columns to snake_case

📋 **REF file:** `REF_dtypes_and_formatting.ipynb`  
📋 **REF file:** `REF_downtyping.ipynb`


```python
# ── Type audit ────────────────────────────────────────────────────────────
# df.dtypes.to_frame('dtype')

# ── Convert numeric (handles $, commas, %) ────────────────────────────────
# df['col'] = df['col'].astype(str).str.replace(r'[$,£€%\s]', '', regex=True)
# df['col'] = pd.to_numeric(df['col'], errors='coerce')

# ── Convert dates ─────────────────────────────────────────────────────────
# df['date'] = pd.to_datetime(df['date'], errors='coerce')
# df['month'] = df['date'].dt.month
# df['dow']   = df['date'].dt.day_name()

# ── Cast to category ──────────────────────────────────────────────────────
# df['cat_col'] = df['cat_col'].astype('category')

# ── Strip whitespace, normalize case ─────────────────────────────────────
# str_cols = df.select_dtypes('object').columns
# df[str_cols] = df[str_cols].apply(lambda x: x.str.strip().str.lower())

# ── Rename columns to snake_case ──────────────────────────────────────────
# df.columns = (df.columns.str.strip().str.lower()
#               .str.replace(' ', '_').str.replace('[^a-z0-9_]', '', regex=True))
```

---
## Step 5 — Detect & Treat Outliers

> *Types must be correct first. Outliers are not automatically errors — understand them before removing.*

**Detection methods:**

| Method | When to use |
|---|---|
| IQR (Tukey fences) | Default. Distribution-agnostic. |
| Z-score (\|z\| > 3) | Assumes normality. Use after confirming distribution. |
| Domain bounds | When you know impossible values (age < 0, age > 120) |
| Isolation Forest | Multivariate outlier detection |

**Treatment options:**

| Treatment | When to use |
|---|---|
| Remove row | Clearly erroneous (age = 999, impossible domain value) |
| Cap / Winsorize | Extreme but plausible; preserve direction |
| Log transform | Right-skewed; retain all values |
| Retain | Outliers are legitimate signal (high-risk loans, power users) |
| Flag | Uncertain — let the model decide |

**Checklist:**
- [ ] Visualize distributions before deciding
- [ ] Apply domain knowledge (what values are impossible?)
- [ ] Document every decision with reasoning
- [ ] Check row count before and after

📋 **REF file:** `REF_outlier_detection.ipynb`  
📋 **Pipeline file:** `TMPL_PIPELINE_outlier_handling.ipynb`


```python
# ── IQR bounds ────────────────────────────────────────────────────────────
# Q1, Q3 = df['col'].quantile([0.25, 0.75])
# IQR = Q3 - Q1
# lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR

# ── Remove ────────────────────────────────────────────────────────────────
# df = df[(df['col'] >= lower) & (df['col'] <= upper)]

# ── Cap (Winsorize) ───────────────────────────────────────────────────────
# df['col'] = df['col'].clip(lower=lower, upper=upper)

# ── Log transform ─────────────────────────────────────────────────────────
# df['col_log'] = np.log1p(df['col'])   # log1p handles zeros

# ── Flag outlier, keep row ────────────────────────────────────────────────
# df['col_is_outlier'] = ((df['col'] < lower) | (df['col'] > upper)).astype(int)

# ── Domain-based removal ──────────────────────────────────────────────────
# df = df[(df['age'] >= 18) & (df['age'] <= 100)]

# ── Z-score flag ──────────────────────────────────────────────────────────
# from scipy import stats
# df['col_zscore'] = np.abs(stats.zscore(df['col'].dropna()))
```

---
## Step 6 — Value Correction & Standardization

> *Inconsistent values silently corrupt groupby operations, encodings, and model inputs.*

**Common issues:**
- Status codes: `'Y'`, `'Yes'`, `'yes'`, `'TRUE'` all meaning the same thing
- Categorical inconsistencies: `'active'`, `'ACTIVE'`, `'Active'`
- Impossible values: `rating = 0` when scale is 1–5, `price = -50`
- Extra whitespace, leading/trailing characters

**Checklist:**
- [ ] Check value_counts() for all categorical columns — look for near-duplicates
- [ ] Standardize text values (case, whitespace, common replacements)
- [ ] Filter rows with impossible numeric values
- [ ] Drop columns with >50% missing (usually)

📋 **REF file:** `REF_dtypes_and_formatting.ipynb`


```python
# ── Standardize categorical values ───────────────────────────────────────
# df['status'] = df['status'].replace({
#     'Y': 'Yes', 'N': 'No', 'active': 'Active', 'ACTIVE': 'Active'
# })

# ── Remove invalid rows ───────────────────────────────────────────────────
# df = df[df['age'] > 0]
# df = df[df['price'] >= 0]
# df = df[df['rating'].between(1, 5)]

# ── Drop columns with excessive nulls ─────────────────────────────────────
# threshold = 0.50
# cols_to_drop = df.columns[df.isnull().mean() > threshold]
# df = df.drop(columns=cols_to_drop)
```

---
## Step 7 — Rename & Reorganize

> *Clean column names prevent silent bugs in downstream code.*

**Checklist:**
- [ ] All columns lowercase snake_case
- [ ] No spaces, special characters, or leading numbers
- [ ] ID columns dropped (if no longer needed)
- [ ] Columns reordered logically (ID → demographics → features → target)


```python
# ── Rename specific columns ───────────────────────────────────────────────
# df = df.rename(columns={
#     'Old Name': 'new_name',
#     'AnotherCol': 'another_col',
# })

# ── Bulk clean all column names ───────────────────────────────────────────
# df.columns = (df.columns.str.lower().str.strip()
#               .str.replace(' ', '_').str.replace('[^a-z0-9_]', '', regex=True))

# ── Reorder columns ───────────────────────────────────────────────────────
# df = df[['id', 'age', 'income', ..., 'target']]
```

---
## Step 8 — Validate & Export

> *Always confirm quality before handing data to the next pipeline stage.*

**Checklist:**
- [ ] Row count vs original (how much did we lose? is that expected?)
- [ ] Zero nulls (or documented intentional nulls)
- [ ] Zero duplicates
- [ ] All dtypes correct
- [ ] No impossible values remain
- [ ] Memory usage reasonable

📋 **Pipeline file:** `TMPL_PIPELINE_validation.ipynb`


```python
# ── Final quality check ───────────────────────────────────────────────────
print(f"✓ Rows:         {len(df):,}")
print(f"✓ Columns:      {df.shape[1]}")
print(f"✓ Nulls:        {df.isnull().sum().sum()}")
print(f"✓ Duplicates:   {df.duplicated().sum()}")
print(f"✓ Memory:       {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# ── Export ────────────────────────────────────────────────────────────────
# df.to_csv('data/processed/cleaned_data.csv', index=False)
# df.to_parquet('data/processed/cleaned_data.parquet', index=False)  # preferred for large files

# ── Preserve a clean backup copy in memory ────────────────────────────────
# df_clean = df.copy()
```
