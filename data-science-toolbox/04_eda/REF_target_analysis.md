# Target Variable Analysis
*Cookbook: `04_eda/`*

Understand your target before modeling.


```python
plt.style.use('ggplot')
PRIMARY_COLOR = 'steelblue'
ALT_COLOR = 'coral'

def analyze_classification_target(df, target, figsize=(8, 5)):
    """
    Analyzes class balance for classification tasks.
    Identifies if resampling (SMOTE) or specialized metrics are needed.
    """
    # Calculation
    vc = df[target].value_counts()
    pct = df[target].value_counts(normalize=True) * 100
    balance_df = pd.DataFrame({'Count': vc, 'Percentage': pct.round(2)})
    
    print(f"--- Target Analysis: {target} ---")
    print(balance_df)
    
    # Visualization
    plt.figure(figsize=figsize)
    ax = vc.plot(kind='bar', color=[PRIMARY_COLOR, ALT_COLOR], edgecolor='white', rot=0)
    
    # Adding labels on top of bars
    for i, v in enumerate(vc):
        ax.text(i, v + (vc.max() * 0.01), f"{pct.iloc[i]:.1f}%", ha='center', fontweight='bold')
        
    plt.title(f'Class Balance: {target}', fontsize=12)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    
    # Heuristic Check for Imbalance
    ratio = vc.max() / vc.min()
    if ratio > 4:
        print(f"\n⚠️ WARNING: Imbalance Ratio is {ratio:.1f}:1.")
        print("Consider using StratifiedKFold, SMOTE, or adjusting 'class_weight'.")

def analyze_regression_target(df, target, bins=50):
    """
    Analyzes distribution and skewness of a continuous target.
    Determines if a log-transform is required to normalize the target.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw Distribution
    sns.histplot(df[target], bins=bins, kde=True, ax=ax1, color=PRIMARY_COLOR, edgecolor='white')
    ax1.set_title(f'Raw {target} (Skew: {df[target].skew():.2f})')
    
    # Log-Transformed Distribution
    log_target = np.log1p(df[target])
    sns.histplot(log_target, bins=bins, kde=True, ax=ax2, color=ALT_COLOR, edgecolor='white')
    ax2.set_title(f'Log1p {target} (Skew: {log_target.skew():.2f})')
    
    plt.tight_layout()
    plt.show()
    
    # Decision logic
    abs_skew = abs(df[target].skew())
    if abs_skew > 1:
        print(f"ℹ️ High Skewness detected ({df[target].skew():.2f}).")
        print("Recommendation: Train model on log-transformed target or use a PowerTransformer.")
```
