# Baseline Models
*Cookbook: `06b_model_development/`*

Always establish a baseline before anything fancy. If you can't beat the dummy, rethink the problem.


```python
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
import numpy as np

# ── Dummy Baselines ───────────────────────────────────────────────────────

# Classification — most frequent class
# dummy_clf = DummyClassifier(strategy='most_frequent')
# dummy_clf.fit(X_train, y_train)
# print(f'Dummy Accuracy: {accuracy_score(y_val, dummy_clf.predict(X_val)):.4f}')

# Regression — predict mean
# dummy_reg = DummyRegressor(strategy='mean')
# dummy_reg.fit(X_train, y_train)
# rmse = np.sqrt(mean_squared_error(y_val, dummy_reg.predict(X_val)))
# print(f'Dummy RMSE: {rmse:.4f}')

```


```python
# ── Linear Baseline ───────────────────────────────────────────────────────

# Logistic Regression (classification)
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression(max_iter=1000, random_state=42)
# lr.fit(X_train_scaled, y_train)
# print(f'LR ROC-AUC: {roc_auc_score(y_val, lr.predict_proba(X_val_scaled)[:,1]):.4f}')

# Ridge Regression (regression)
# from sklearn.linear_model import Ridge
# ridge = Ridge(alpha=1.0)
# ridge.fit(X_train_scaled, y_train)
# rmse = np.sqrt(mean_squared_error(y_val, ridge.predict(X_val_scaled)))
# print(f'Ridge RMSE: {rmse:.4f}')

```
