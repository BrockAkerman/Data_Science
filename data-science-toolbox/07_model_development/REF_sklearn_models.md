# Sklearn Model Library
*Cookbook: `06b_model_development/`*

Common model setups with sensible defaults.


```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb

SEED = 42

# ── Classification Models ─────────────────────────────────────────────────
models_clf = {
    'logistic':   LogisticRegression(max_iter=1000, random_state=SEED),
    'rf':         RandomForestClassifier(n_estimators=200, max_depth=None, random_state=SEED, n_jobs=-1),
    'xgb':        xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, use_label_encoder=False,
                                      eval_metric='logloss', random_state=SEED),
    'lgb':        lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=SEED, verbose=-1),
}

# ── Regression Models ─────────────────────────────────────────────────────
models_reg = {
    'ridge':      Ridge(alpha=1.0),
    'rf':         RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
    'xgb':        xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=SEED),
    'lgb':        lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=SEED, verbose=-1),
}

```


```python
# ── Quick Comparison Loop (classification) ───────────────────────────────
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd

def compare_models_clf(models, X_train, y_train, X_val, y_val):
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
        results.append({
            'model': name,
            'accuracy': accuracy_score(y_val, preds),
            'roc_auc': roc_auc_score(y_val, proba) if proba is not None else None
        })
    return pd.DataFrame(results).sort_values('roc_auc', ascending=False)

# display(compare_models_clf(models_clf, X_train, y_train, X_val, y_val))

```


```python
# ── Hyperparameter Tuning: GridSearchCV ──────────────────────────────────
from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth':    [3, 5, None],
#     'min_samples_split': [2, 5]
# }
# grid = GridSearchCV(models_clf['rf'], param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
# grid.fit(X_train, y_train)
# print(f'Best params: {grid.best_params_}')
# print(f'Best CV score: {grid.best_score_:.4f}')
# best_model = grid.best_estimator_

```
