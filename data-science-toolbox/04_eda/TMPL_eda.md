# 📊 Template: Exploratory Data Analysis (EDA)

**Purpose:** Reusable visualization functions for understanding your dataset before modeling.  
Assumes a cleaned `df` DataFrame and a binary `target_col`.  

**Sections:**
1. Summary Statistics
2. Target Distribution
3. Univariate Analysis (histograms, boxplots, ECDFs)
4. Bivariate Analysis (default rate by feature)
5. Correlation Heatmap
6. Class Separation (boxplots by target)



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

sns.set_theme(style='whitegrid', context='talk', palette='deep')

# CHANGE THESE
df         = pd.read_csv('your_cleaned_data.csv')   # your cleaned dataframe
TARGET_COL = 'target'                                # your binary target column (0/1)

print(f'Shape: {df.shape}')
df.head(3)

```

## 1️⃣  Summary Statistics


```python
# Numeric summary
num_cols = df.select_dtypes(include='number').columns.tolist()
cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()

print(f'Numeric columns  ({len(num_cols)}): {num_cols}')
print(f'Categorical cols ({len(cat_cols)}): {cat_cols}')
print()
df[num_cols].describe().round(2)

```

## 2️⃣  Target Distribution


```python
def plot_target_distribution(df, target_col):
    counts    = df[target_col].value_counts()
    rates     = df[target_col].value_counts(normalize=True) * 100
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=100)

    axes[0].bar(counts.index.astype(str), counts.values, color=['#1a434e','#e74c3c'], edgecolor='white')
    axes[0].set_title('Class Counts', fontweight='bold')
    axes[0].set_xlabel(target_col)
    axes[0].set_ylabel('Count')

    axes[1].pie(rates.values, labels=[f'Class {i}\n({v:.1f}%)' for i, v in rates.items()],
                colors=['#1a434e','#e74c3c'], startangle=90, wedgeprops={'edgecolor':'white','linewidth':2})
    axes[1].set_title('Class Balance', fontweight='bold')

    sns.despine()
    plt.tight_layout()
    plt.show()

    print(f'Target distribution:')
    for cls, count in counts.items():
        print(f'  Class {cls}: {count:,}  ({rates[cls]:.1f}%)')

plot_target_distribution(df, TARGET_COL)

```

## 3️⃣  Univariate Analysis

Run for each numeric feature you want to inspect.


```python
def plot_histogram(df, col, bins=30, kde=True):
    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
    sns.histplot(df[col].dropna(), bins=bins, kde=kde, ax=ax,
                 color='#1a434e', edgecolor='white')
    ax.set_title(f'Distribution: {col}', fontweight='bold', loc='left')
    ax.set_xlabel(col)
    sns.despine()
    plt.tight_layout()
    plt.show()
    print(df[col].describe().round(4))
    print(f'Skewness: {df[col].skew():.4f}')

def plot_boxplot(df, col):
    fig, ax = plt.subplots(figsize=(10, 3), dpi=100)
    sns.boxplot(x=df[col].dropna(), ax=ax, color='#1a434e')
    ax.set_title(f'Boxplot: {col}', fontweight='bold', loc='left')
    sns.despine()
    plt.tight_layout()
    plt.show()

# CHANGE: call for each feature you want to inspect
# plot_histogram(df, 'your_numeric_col')
# plot_boxplot(df, 'your_numeric_col')

# Quick loop over all numeric columns:
for col in num_cols[:5]:    # change [:5] to slice as needed
    plot_histogram(df, col)

```

## 4️⃣  Bivariate Analysis — Default Rate by Feature

Shows how the target rate varies across categories or bins.


```python
def plot_default_rate_by_feature(df, feature, target_col, max_cats=15):
    """
    For categorical features: bar chart of target rate per category.
    For high-cardinality / numeric features: bin into deciles first.
    """
    if df[feature].dtype.name in ('object', 'category') and df[feature].nunique() <= max_cats:
        grp = df.groupby(feature)[target_col].mean().sort_values(ascending=False)
    else:
        # Bin into 10 deciles
        binned = pd.qcut(df[feature], q=10, duplicates='drop')
        grp = df.groupby(binned)[target_col].mean()

    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
    bars = ax.bar(grp.index.astype(str), grp.values * 100,
                  color='#1a434e', edgecolor='white')
    ax.axhline(y=df[target_col].mean() * 100, color='#e74c3c', ls='--', lw=1.5,
               label=f'Overall rate ({df[target_col].mean():.1%})')
    ax.set_ylabel('Default Rate (%)', fontweight='bold')
    ax.set_title(f'Target Rate by {feature}', fontweight='bold', loc='left')
    ax.tick_params(axis='x', rotation=30)
    ax.legend()
    sns.despine()
    plt.tight_layout()
    plt.show()

# CHANGE: call for each feature you want to inspect
# plot_default_rate_by_feature(df, 'home_type', TARGET_COL)
# plot_default_rate_by_feature(df, 'age', TARGET_COL)

for col in cat_cols[:3]:   # quick loop over first 3 categorical cols
    plot_default_rate_by_feature(df, col, TARGET_COL)

```

## 5️⃣  Correlation Heatmap


```python
fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Matrix (Numeric Features)', fontweight='bold', loc='left')
plt.tight_layout()
plt.show()

```

## 6️⃣  Class Separation — Boxplots


```python
def plot_class_separation(df, features, target_col, n_cols=3):
    """Boxplots of numeric features split by target class."""
    n_rows = -(-len(features) // n_cols)   # ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), dpi=100)
    axes = axes.flatten()

    for i, feat in enumerate(features):
        sns.boxplot(data=df, x=target_col, y=feat, ax=axes[i],
                    palette={0: '#1a434e', 1: '#e74c3c'})
        axes[i].set_title(feat, fontweight='bold')
        axes[i].set_xlabel('Target (0=Neg, 1=Pos)')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Feature Distribution by Target Class', fontweight='bold', fontsize=14)
    sns.despine()
    plt.tight_layout()
    plt.show()

# CHANGE: list the numeric features you want to compare by class
plot_class_separation(df, num_cols[:6], TARGET_COL)

```
