# Univariate Analysis
*Cookbook: `04_eda/`*

Distributions for numeric and categorical columns.


```python
# Standard Visual Settings
plt.style.use('ggplot') # Or your preferred clean style
PRIMARY_COLOR = 'steelblue'

def plot_numeric_distributions(df, cols=None, bins=30, figsize_per=(4, 3)):
    """
    Generates a grid of histograms for numeric columns.
    Modular: Handles any number of columns by auto-calculating grid rows.
    """
    # Auto-select numeric columns if none provided
    cols = cols or df.select_dtypes(include=np.number).columns.tolist()
    if not cols:
        print("No numeric columns found.")
        return

    n = len(cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per[0]*ncols, figsize_per[1]*nrows))
    
    # Flatten axes for easy iteration, handle single-subplot case
    axes_list = axes.flatten() if n > 1 else [axes]
    
    for i, col in enumerate(cols):
        ax = axes_list[i]
        df[col].hist(ax=ax, bins=bins, edgecolor='white', color=PRIMARY_COLOR)
        ax.set_title(f"Dist: {col}", fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
    # Hide unused subplots
    for j in range(i + 1, len(axes_list)):
        axes_list[j].set_visible(False)
        
    fig.tight_layout()
    plt.show()

def get_numeric_summary(df):
    """
    Returns a comprehensive statistical summary including shape and nulls.
    Useful for quick validation during technical drills.
    """
    num_df = df.select_dtypes(include=np.number)
    if num_df.empty:
        return "No numeric data available."
        
    summary = num_df.describe().T
    summary['skew'] = num_df.skew()
    summary['kurtosis'] = num_df.kurtosis()
    summary['null_count'] = num_df.isnull().sum()
    summary['null_pct'] = (num_df.isnull().mean() * 100).round(2)
    
    # Reordering for readability: Mean/Std/Nulls first
    cols_order = ['count', 'null_count', 'null_pct', 'mean', 'std', 'min', '50%', 'max', 'skew']
    return summary[cols_order]

def plot_categorical_distributions(df, cols=None, top_n=10, figsize=(8, 4)):
    """
    Generates horizontal bar charts for categorical columns.
    Intermodular: Use this to identify rare labels for feature engineering.
    """
    cols = cols or df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not cols:
        print("No categorical columns found.")
        return

    for col in cols:
        # Calculate percentages for labels
        vc = df[col].value_counts().head(top_n)
        total = len(df[col].dropna())
        
        plt.figure(figsize=figsize)
        ax = vc.plot(kind='barh', color=PRIMARY_COLOR, edgecolor='white')
        
        # Add frequency labels to bars
        for i, v in enumerate(vc):
            pct = f"{(v/total)*100:.1f}%"
            ax.text(v + (vc.max()*0.01), i, f"{v} ({pct})", va='center', fontsize=9)
            
        plt.title(f'{col} — Top {top_n} (Unique: {df[col].nunique()})', fontsize=12)
        plt.gca().invert_yaxis()
        plt.xlabel("Frequency")
        plt.tight_layout()
        plt.show()
```
