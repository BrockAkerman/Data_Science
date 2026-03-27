# Feature Interaction & Non-Linearity
*Cookbook: `04_eda/`*

Standard correlation heatmaps only find linear relationships. This module uses Mutual Information to find non-linear connections.


```python
def plot_mutual_information(df, target, task='classification', top_n=15):
    """
    Calculates and plots Mutual Information scores.
    Identifies non-linear relationships that Correlation Heatmaps miss.
    """
    X = df.select_dtypes(include=np.number).drop(columns=[target]).fillna(0)
    y = df[target]
    
    mi_func = mutual_info_classif if task == 'classification' else mutual_info_regression
    mi_scores = mi_func(X, y, random_state=42)
    
    mi_df = pd.Series(mi_scores, name="MI Scores", index=X.columns).sort_values(ascending=False)
    
    plt.figure(figsize=(10, top_n * 0.4))
    mi_df.head(top_n).plot(kind='barh', color='teal')
    plt.title(f'Mutual Information Scores vs {target}')
    plt.xlabel('Information Gain')
    plt.gca().invert_yaxis()
    plt.show()
    
    return mi_df

def plot_interaction_discovery(df, x, y, hue, figsize=(10, 6)):
    """
    Visualizes how a third variable affects the relationship between two others.
    Essential for identifying feature 'synergy'.
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=0.6, palette='viridis')
    plt.title(f'Interaction: {x} vs {y} by {hue}')
    plt.show()
```
