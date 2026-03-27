# Temporal Stability & Drift Analysis
*Cookbook: `04_eda/`*

This module is for investigating how feature distributions change over time, which is critical for preventing model decay.


```python
def plot_feature_drift(df, date_col, feature_cols, freq='M'):
    """
    Plots feature means over time to identify 'Data Drift'.
    If the line isn't flat, your training data might not represent the future.
    """
    temp_df = df.copy()
    temp_df[date_col] = pd.to_datetime(temp_df[date_col])
    
    for col in feature_cols:
        # Resample to get time-based mean and std
        resampled = temp_df.set_index(date_col)[col].resample(freq).agg(['mean', 'std'])
        
        plt.figure(figsize=(12, 4))
        plt.plot(resampled.index, resampled['mean'], marker='o', color='steelblue', label='Mean')
        plt.fill_between(resampled.index, 
                         resampled['mean'] - resampled['std'], 
                         resampled['mean'] + resampled['std'], 
                         color='steelblue', alpha=0.2, label='Std Dev')
        
        plt.title(f'Temporal Drift: {col} (Agg: {freq})')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

def plot_target_stability(df, date_col, target, freq='W'):
    """
    Analyzes if the target rate (e.g., churn, default) changes over time.
    """
    temp_df = df.copy()
    temp_df[date_col] = pd.to_datetime(temp_df[date_col])
    
    stability = temp_df.set_index(date_col)[target].resample(freq).mean()
    
    plt.figure(figsize=(12, 4))
    stability.plot(kind='line', color='coral', linewidth=2, marker='s')
    plt.axhline(stability.mean(), color='black', linestyle='--', alpha=0.5)
    plt.title(f'Target Stability Over Time ({target})')
    plt.ylabel('Mean Target Rate')
    plt.show()
```
