# Bivariate Analysis
*Cookbook: `04_eda/`*

Relationships between pairs of features and vs. target.


```python
plt.style.use('ggplot')
PRIMARY_COLOR = 'steelblue'
ALT_COLOR = 'orange'

def plot_correlation_heatmap(df, cols=None, figsize=(10, 8)):
    """
    Plots a triangular correlation heatmap. 
    Essential for identifying multicollinearity before modeling.
    """
    cols = cols or df.select_dtypes(include=np.number).columns.tolist()
    if not cols: return print("No numeric columns.")
    
    corr = df[cols].corr()
    # Mask the upper triangle for clarity
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, linewidths=0.5,
                annot_kws={'size': 9})
    plt.title('Numeric Correlation Matrix (Triangular)', fontsize=14, pad=20)
    plt.show()

def plot_numeric_vs_target(df, target, cols=None, figsize_per=(5, 4)):
    """
    Plots distributions of numeric features segmented by the target.
    Works for binary classification or low-cardinality categorical targets.
    """
    cols = cols or df.select_dtypes(include=np.number).columns.difference([target]).tolist()
    n = len(cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per[0]*ncols, figsize_per[1]*nrows))
    axes_list = axes.flatten() if n > 1 else [axes]
    
    for i, col in enumerate(cols):
        ax = axes_list[i]
        for val in sorted(df[target].dropna().unique()):
            sns.kdeplot(df[df[target] == val][col], ax=ax, fill=True, label=f"Target: {val}", alpha=0.4)
        ax.set_title(f'{col} vs {target}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        
    for j in range(i + 1, len(axes_list)): axes_list[j].set_visible(False)
    fig.tight_layout()
    plt.show()

def plot_reg_scatter(df, x, y, hue=None):
    """
    Standard regression plot with optional hue. 
    Use to identify linear relationships and outliers.
    """
    plt.figure(figsize=(8, 5))
    if hue:
        sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=0.6)
    else:
        sns.regplot(data=df, x=x, y=y, scatter_kws={'alpha':0.4, 'color': PRIMARY_COLOR}, 
                    line_kws={'color': ALT_COLOR})
    plt.title(f'Relationship: {x} vs {y}', fontsize=12)
    plt.show()
```
