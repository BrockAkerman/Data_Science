# Target Variable Analysis
*Cookbook: `04_eda/`*

Understand your target before modeling.


```python
def plot_missing_matrix(df, figsize=(12, 6)):
    """
    Visualizes the location of missing values across the dataset.
    Helps identify if nulls are grouped by rows or occur in specific blocks.
    """
    plt.figure(figsize=figsize)
    # Using a binary heatmap to show null positions
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title('Missing Data Matrix (Yellow = Missing)', fontsize=14)
    plt.show()

def plot_missing_correlations(df):
    """
    Identifies if the absence of one variable correlates with the absence of another.
    Crucial for identifying systematic data collection failures.
    """
    # Create a shadow dataframe where 1 is null and 0 is not
    null_corr = df.isnull().corr()
    # Filter to show only columns that actually have nulls
    null_corr = null_corr.dropna(how='all', axis=0).dropna(how='all', axis=1)
    
    if null_corr.empty:
        return print("No missing value correlations to display.")
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(null_corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Missingness Correlation (Do variables go missing together?)')
    plt.show()

def run_missing_eda(df):
    """
    Integrates your existing diagnostic logic with visuals.
    """
    # Using your 'null_diagnostic' logic to find hidden nulls
    hidden_patterns = ['?', 'n/a', 'nan', 'none', '', ' ', 'null']
    
    print("--- Missingness Audit ---")
    plot_missing_matrix(df)
    plot_missing_correlations(df)
    
    # Text summary of your diagnostic logic
    null_counts = df.isnull().sum()
    null_pct = (df.isnull().mean() * 100).round(2)
    print(pd.DataFrame({'Null Count': null_counts, 'Pct': null_pct}).query('Null Count > 0'))
```
