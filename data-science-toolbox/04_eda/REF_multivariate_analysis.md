# Multivariate Analysis
*Cookbook: `04_eda/`*

Relationships between pairs of features and vs. target.


```python
def plot_pairgrid(df, cols, hue=None, max_cols=6):
    """
    Standard PairPlot with safeguards. 
    Limited to max_cols to prevent performance issues.
    """
    if len(cols) > max_cols:
        print(f"Trimming plot from {len(cols)} to first {max_cols} columns for performance.")
        cols = cols[:max_cols]
    
    plot_df = df[cols + ([hue] if hue else [])].dropna()
    g = sns.pairplot(plot_df, hue=hue, corner=True, diag_kind='kde', 
                     plot_kws={'alpha': 0.5, 's': 30})
    g.fig.suptitle('Multivariate Pair Grid', y=1.02, fontsize=14)
    plt.show()

def plot_facet_grid(df, x, y, row=None, col=None, hue=None):
    """
    Powerful tool for seeing how relationships change across categories.
    Example: plot_facet_grid(df, x='Weight', y='Height', col='Gender', row='AgeGroup')
    """
    g = sns.FacetGrid(df, row=row, col=col, hue=hue, height=3.5, aspect=1.2)
    g.map(sns.scatterplot, x, y, alpha=0.6)
    g.add_legend()
    plt.show()

def plot_parallel_coordinates(df, cols, class_col):
    """
    Visualizes high-dimensional data profiles. 
    Especially useful for identifying clusters in classification tasks.
    """
    from pandas.plotting import parallel_coordinates
    
    plt.figure(figsize=(12, 6))
    # Standardize data for visual comparison if scales differ
    plot_df = df[cols + [class_col]].copy()
    for col in cols:
        plot_df[col] = (plot_df[col] - plot_df[col].mean()) / plot_df[col].std()
        
    parallel_coordinates(plot_df, class_col, color=sns.color_palette("viridis", df[class_col].nunique()))
    plt.title('Standardized Parallel Coordinates Plot', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(alpha=0.2)
    plt.show()
```
