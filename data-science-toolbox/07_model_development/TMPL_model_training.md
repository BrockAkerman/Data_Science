# 🤖 Template: Model Training — All Classifiers

**Purpose:** Train, compare, and select a champion classifier.  
Includes: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, LightGBM, XGBoost, plus RandomizedSearchCV tuning for each boosting model.

**Assumes:** `X_train`, `X_val`, `y_train`, `y_val` exist from the split notebook.

**How to use:**
1. Run Setup.
2. Run the model blocks you want (each is independent).
3. Run the Comparison block to pick your champion.



```python
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb
import xgboost  as xgb

from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics         import roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics         import roc_curve
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
CV_FOLDS     = 5
cv           = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Model registry — stores all trained models for comparison
model_registry = {}

# ── Pipeline builder ──────────────────────────────────────────────────────────
def build_pipeline(estimator) -> Pipeline:
    """Wrap any estimator in StandardScaler → Model pipeline."""
    return Pipeline([('scaler', StandardScaler()), ('model', estimator)])

# ── Register helper ───────────────────────────────────────────────────────────
def register_model(name, pipeline, cv_scores, val_proba):
    val_auc = roc_auc_score(y_val, val_proba)
    model_registry[name] = {
        'pipeline'    : pipeline,
        'cv_auc_mean' : cv_scores.mean(),
        'cv_auc_std'  : cv_scores.std(),
        'val_auc'     : val_auc,
        'val_proba'   : val_proba,
    }
    print(f'  {name:<40} CV: {cv_scores.mean():.4f} ±{cv_scores.std():.4f}  |  Val AUC: {val_auc:.4f}')

print(f'Setup complete. CV: Stratified {CV_FOLDS}-fold  |  Random state: {RANDOM_STATE}')

```

## 1️⃣  Logistic Regression (baseline)


```python
# ── Logistic Regression ───────────────────────────────────────────────────────
# CHANGE: adjust C for regularisation strength (smaller = stronger regularisation)
lr_pipe = build_pipeline(
    LogisticRegression(
        class_weight = 'balanced',
        max_iter     = 1000,
        solver       = 'lbfgs',
        C            = 1.0,          # CHANGE
        random_state = RANDOM_STATE,
    )
)
lr_pipe.fit(X_train, y_train)
lr_cv    = cross_val_score(lr_pipe, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
lr_proba = lr_pipe.predict_proba(X_val)[:, 1]

print('Logistic Regression')
print('='*60)
register_model('Logistic Regression', lr_pipe, lr_cv, lr_proba)
print()
print(classification_report(y_val, lr_pipe.predict(X_val)))

```

## 2️⃣  Decision Tree


```python
# ── Decision Tree ─────────────────────────────────────────────────────────────
# CHANGE: max_depth controls complexity; min_samples_leaf prevents overfitting
dt_pipe = build_pipeline(
    DecisionTreeClassifier(
        class_weight     = 'balanced',
        max_depth        = 6,        # CHANGE
        min_samples_leaf = 20,       # CHANGE
        random_state     = RANDOM_STATE,
    )
)
dt_pipe.fit(X_train, y_train)
dt_cv    = cross_val_score(dt_pipe, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
dt_proba = dt_pipe.predict_proba(X_val)[:, 1]

print('Decision Tree')
print('='*60)
register_model('Decision Tree', dt_pipe, dt_cv, dt_proba)
print()
print(classification_report(y_val, dt_pipe.predict(X_val)))

```

## 3️⃣  Random Forest


```python
# ── Random Forest ─────────────────────────────────────────────────────────────
# CHANGE: n_estimators, max_depth, min_samples_leaf
rf_pipe = build_pipeline(
    RandomForestClassifier(
        n_estimators     = 300,      # CHANGE
        max_features     = 'sqrt',
        max_depth        = 12,       # CHANGE
        min_samples_leaf = 10,       # CHANGE
        class_weight     = 'balanced',
        random_state     = RANDOM_STATE,
        n_jobs           = -1,
    )
)
rf_pipe.fit(X_train, y_train)
rf_cv    = cross_val_score(rf_pipe, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
rf_proba = rf_pipe.predict_proba(X_val)[:, 1]

print('Random Forest')
print('='*60)
register_model('Random Forest', rf_pipe, rf_cv, rf_proba)
print()
print(classification_report(y_val, rf_pipe.predict(X_val)))

```

## 4️⃣  Gradient Boosting (sklearn)


```python
# ── Gradient Boosting ─────────────────────────────────────────────────────────
# CHANGE: n_estimators, learning_rate, max_depth
gb_pipe = build_pipeline(
    GradientBoostingClassifier(
        n_estimators  = 300,         # CHANGE
        learning_rate = 0.05,        # CHANGE (lower = slower but usually better)
        max_depth     = 4,           # CHANGE
        subsample     = 0.8,
        max_features  = 'sqrt',
        random_state  = RANDOM_STATE,
    )
)
gb_pipe.fit(X_train, y_train)
gb_cv    = cross_val_score(gb_pipe, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
gb_proba = gb_pipe.predict_proba(X_val)[:, 1]

print('Gradient Boosting')
print('='*60)
register_model('Gradient Boosting', gb_pipe, gb_cv, gb_proba)
print()
print(classification_report(y_val, gb_pipe.predict(X_val)))

```

## 5️⃣  LightGBM


```python
# ── LightGBM ──────────────────────────────────────────────────────────────────
# CHANGE: key hyperparameters annotated below
lgbm_pipe = build_pipeline(
    lgb.LGBMClassifier(
        n_estimators     = 300,      # CHANGE: number of boosting rounds
        learning_rate    = 0.05,     # CHANGE: lower = better generalization, slower
        max_depth        = 6,        # CHANGE: tree depth; -1 = unlimited
        num_leaves       = 31,       # CHANGE: max leaves per tree (2^max_depth suggestion)
        subsample        = 0.8,      # row subsampling per tree
        colsample_bytree = 0.8,      # feature subsampling per tree
        is_unbalance     = True,     # native class imbalance handling
        random_state     = RANDOM_STATE,
        n_jobs           = -1,
        verbose          = -1,
    )
)
lgbm_pipe.fit(X_train, y_train)
lgbm_cv    = cross_val_score(lgbm_pipe, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
lgbm_proba = lgbm_pipe.predict_proba(X_val)[:, 1]

print('LightGBM')
print('='*60)
register_model('LightGBM', lgbm_pipe, lgbm_cv, lgbm_proba)
print()
print(classification_report(y_val, lgbm_pipe.predict(X_val)))

```

## 6️⃣  XGBoost


```python
# ── XGBoost ───────────────────────────────────────────────────────────────────
# scale_pos_weight = negatives / positives (imbalance ratio)
neg = int((y_train == 0).sum())
pos = int((y_train == 1).sum())
spw = neg / pos
print(f'scale_pos_weight: {spw:.2f}  ({neg:,} neg / {pos:,} pos)')

xgb_pipe = build_pipeline(
    xgb.XGBClassifier(
        n_estimators     = 300,      # CHANGE
        learning_rate    = 0.05,     # CHANGE
        max_depth        = 5,        # CHANGE
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = spw,      # handles class imbalance
        reg_alpha        = 0.1,      # L1 regularisation
        reg_lambda       = 1.0,      # L2 regularisation
        random_state     = RANDOM_STATE,
        n_jobs           = -1,
        eval_metric      = 'auc',
        verbosity        = 0,
    )
)
xgb_pipe.fit(X_train, y_train)
xgb_cv    = cross_val_score(xgb_pipe, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
xgb_proba = xgb_pipe.predict_proba(X_val)[:, 1]

print('XGBoost')
print('='*60)
register_model('XGBoost', xgb_pipe, xgb_cv, xgb_proba)
print()
print(classification_report(y_val, xgb_pipe.predict(X_val)))

```

## 7️⃣  Hyperparameter Tuning — RandomizedSearchCV


```python
# ── Tune: Random Forest ───────────────────────────────────────────────────────
rf_param_dist = {
    'model__n_estimators'     : [100, 200, 300, 500],
    'model__max_depth'        : [6, 8, 10, 12, None],
    'model__min_samples_leaf' : [5, 10, 20, 50],
    'model__max_features'     : ['sqrt', 'log2', 0.5],
}
rf_search = RandomizedSearchCV(
    build_pipeline(RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)),
    param_distributions=rf_param_dist, n_iter=50, cv=cv, scoring='roc_auc',
    refit=True, n_jobs=-1, random_state=RANDOM_STATE, verbose=1
)
rf_search.fit(X_train, y_train)
print(f'RF best params : {rf_search.best_params_}')
print(f'RF best CV AUC : {rf_search.best_score_:.4f}')
rf_tuned_cv    = cross_val_score(rf_search.best_estimator_, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
rf_tuned_proba = rf_search.best_estimator_.predict_proba(X_val)[:, 1]
register_model('Random Forest (Tuned)', rf_search.best_estimator_, rf_tuned_cv, rf_tuned_proba)

```


```python
# ── Tune: LightGBM ────────────────────────────────────────────────────────────
lgbm_param_dist = {
    'model__n_estimators'     : [200, 300, 500, 700],
    'model__learning_rate'    : stats.loguniform(0.01, 0.2),
    'model__max_depth'        : [4, 6, 8, 10, -1],
    'model__num_leaves'       : [20, 31, 50, 63, 80],
    'model__subsample'        : stats.uniform(0.6, 0.4),
    'model__colsample_bytree' : stats.uniform(0.6, 0.4),
    'model__min_child_samples': [10, 20, 30, 50],
}
lgbm_search = RandomizedSearchCV(
    build_pipeline(lgb.LGBMClassifier(is_unbalance=True, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
    param_distributions=lgbm_param_dist, n_iter=50, cv=cv, scoring='roc_auc',
    refit=True, n_jobs=-1, random_state=RANDOM_STATE, verbose=1
)
lgbm_search.fit(X_train, y_train)
print(f'LGBM best params : {lgbm_search.best_params_}')
print(f'LGBM best CV AUC : {lgbm_search.best_score_:.4f}')
lgbm_tuned_cv    = cross_val_score(lgbm_search.best_estimator_, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
lgbm_tuned_proba = lgbm_search.best_estimator_.predict_proba(X_val)[:, 1]
register_model('LightGBM (Tuned)', lgbm_search.best_estimator_, lgbm_tuned_cv, lgbm_tuned_proba)

```


```python
# ── Tune: XGBoost ─────────────────────────────────────────────────────────────
xgb_param_dist = {
    'model__n_estimators'     : [200, 300, 500, 700],
    'model__learning_rate'    : stats.loguniform(0.01, 0.2),
    'model__max_depth'        : [3, 4, 5, 6, 8],
    'model__subsample'        : stats.uniform(0.6, 0.4),
    'model__colsample_bytree' : stats.uniform(0.6, 0.4),
    'model__reg_alpha'        : stats.loguniform(0.01, 1.0),
    'model__reg_lambda'       : stats.loguniform(0.5, 5.0),
    'model__scale_pos_weight' : [spw],
}
xgb_search = RandomizedSearchCV(
    build_pipeline(xgb.XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric='auc', verbosity=0)),
    param_distributions=xgb_param_dist, n_iter=50, cv=cv, scoring='roc_auc',
    refit=True, n_jobs=-1, random_state=RANDOM_STATE, verbose=1
)
xgb_search.fit(X_train, y_train)
print(f'XGB best params : {xgb_search.best_params_}')
print(f'XGB best CV AUC : {xgb_search.best_score_:.4f}')
xgb_tuned_cv    = cross_val_score(xgb_search.best_estimator_, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
xgb_tuned_proba = xgb_search.best_estimator_.predict_proba(X_val)[:, 1]
register_model('XGBoost (Tuned)', xgb_search.best_estimator_, xgb_tuned_cv, xgb_tuned_proba)

```

## 8️⃣  Model Comparison & Champion Selection


```python
# ── Comparison table ──────────────────────────────────────────────────────────
rows = [{'Model': name, 'CV AUC': round(e['cv_auc_mean'],4),
         'CV Std': round(e['cv_auc_std'],4), 'Val AUC': round(e['val_auc'],4)}
        for name, e in model_registry.items()]
comparison_df = (pd.DataFrame(rows).sort_values('Val AUC', ascending=False).reset_index(drop=True))
comparison_df.index += 1
print('='*65)
print('MODEL COMPARISON — Validation AUC-ROC')
print('='*65)
print(comparison_df.to_string())

CHAMPION_NAME     = comparison_df.iloc[0]['Model']
CHAMPION_PIPELINE = model_registry[CHAMPION_NAME]['pipeline']
print(f'\n🏆 Champion: {CHAMPION_NAME}  (Val AUC = {model_registry[CHAMPION_NAME]["val_auc"]:.4f})')

# ── ROC curves ────────────────────────────────────────────────────────────────
palette = ['#1a434e','#e74c3c','#2980b9','#27ae60','#8e44ad','#e67e22','#16a085','#c0392b']
fig, ax = plt.subplots(figsize=(10,7), dpi=100)
for idx, (name, e) in enumerate(model_registry.items()):
    fpr, tpr, _ = roc_curve(y_val, e['val_proba'])
    lw = 2.5 if name == CHAMPION_NAME else 1.5
    ls = '-'  if name == CHAMPION_NAME else '--'
    ax.plot(fpr, tpr, lw=lw, ls=ls, color=palette[idx % len(palette)],
            label=f'{name}  (AUC={e["val_auc"]:.4f})')
ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.4,label='Random')
ax.set_xlabel('False Positive Rate', fontweight='bold')
ax.set_ylabel('True Positive Rate',  fontweight='bold')
ax.set_title('ROC Curves — Validation Set', fontweight='bold', loc='left')
ax.legend(loc='lower right', fontsize=9)
sns.despine()
plt.tight_layout()
plt.show()

```
