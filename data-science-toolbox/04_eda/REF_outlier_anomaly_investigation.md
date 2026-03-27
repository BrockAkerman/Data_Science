# Outlier Anomaly Investigation
*Cookbook: `04_eda/`*

This module moves beyond simple boxplots to provide statistical detection of potential "bad data" versus "signal."


```python
plt.style.use('ggplot')
PRIMARY_COLOR = 'steelblue'
OUTLIER_COLOR = 'crimson'

def get_outlier_summary(df, cols=None, threshold=3):
    """
    Identifies outliers using Z-score and IQR methods.
    Returns a summary table of outlier counts per column.
    """
    cols = cols or df.select_dtypes(include=np.number).columns.tolist()
    summary = []
    
    for col in cols:
        series = df[col].dropna()
        # Z-Score
        z_scores = (series - series.mean()) / series.std()
        z_outliers = (np.abs(z_scores) > threshold).sum()
        
        # IQR
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        iqr_outliers = ((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).sum()
        
        summary.append({
            'column': col,
            'z_score_outliers': z_outliers,
            'iqr_outliers': iqr_outliers,
            'outlier_pct': round((iqr_outliers / len(df)) * 100, 2)
        })
    
    return pd.DataFrame(summary).sort_values('iqr_outliers', ascending=False)

def plot_outlier_investigation(df, col):
    """
    Triple-view: Boxplot, Histogram, and Isolation Forest anomaly detection.
    Helps decide if an outlier is 'error' or 'extreme signal'.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Standard Boxplot
    sns.boxplot(x=df[col], ax=axes[0], color=PRIMARY_COLOR)
    axes[0].set_title(f'Boxplot: {col}')
    
    # 2. Histogram with threshold lines
    sns.histplot(df[col], kde=True, ax=axes[1], color=PRIMARY_COLOR)
    axes[1].axvline(df[col].mean(), color='black', linestyle='--', label='Mean')
    axes[1].set_title(f'Distribution: {col}')
    
    # 3. Isolation Forest Anomaly Score
    model = IsolationForest(contamination=0.05, random_state=42)
    # Reshape for sklearn and drop nulls for the fit
    data_clean = df[[col]].dropna()
    preds = model.fit_predict(data_clean)
    
    # Plotting points colored by anomaly status
    axes[2].scatter(range(len(data_clean)), data_clean[col], 
                    c=['red' if p == -1 else 'steelblue' for p in preds], alpha=0.5)
    axes[2].set_title('Isolation Forest (Red = Anomaly)')
    
    plt.tight_layout()
    plt.show()
```
