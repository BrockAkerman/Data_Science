# Naive Bayes

## Overview

## Naive Bayes implementation


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
```

#### Gaussian Naive Bayes


```python
from sklearn.naive_bayes import GaussianNB
```

Checklist:  
-- Read in the Data  
-- Check for class imbalance for dependent variables  
-- If class imbalance absent, procede; otherwise consider correcting the imbalance
-- Works best when conditionally independent.


```python
# Class Imbalance Check
data[''].value_counts()
```


```python
# Define X and y
X = data.drop(columns=['target'])
y = data['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,  # Partition
    stratify=y,  # Stratify keeps the balance the same proportionally for both sets of data. 
    random_state=42  # Seeding
)

# Fit model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Predictions
y_preds = gnb.predict(X_test)
y_probs = gnb.predict_proba(X_test)[:, 1]

```


```python
# Evaluation
print("Accuracy:", accuracy_score(y_test, y_preds))  # Accuracy: <70%=often weak, 75-85%=Solid, 85-95%=Strong, >95%=Excellent/dataset imbalanced
print("Precision:", precision_score(y_test, y_preds))  # Precision:  <0.6=Many False Alarms, 0.7-0.8=Reasonable, 0.8-0.9=Strong, >0.9=Very Reliable
print("Recall:", recall_score(y_test, y_preds))  # Recall:  <0.6=Missing too many cases, 0.7-0.8=Acceptable, 0.8-0.9=Strong Detection, >0.9=Very Sensitive Model
print("F1:", f1_score(y_test, y_preds))  # F1 Score:  <0.6=Weak, 0.7-0.8=Good, 0.8-0.9=Very Good, >0.9=Excellent

print("\nDetailed Report:")
print(classification_report(y_test, y_preds))

print("Baseline:", y_test.value_counts(normalize=True).max())
print("ROC AUC:", roc_auc_score(y_test, y_probs))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_preds))

# Note:  If any of these metrics return zero, sometimes but not always, it might be a scaling issue between the one more more independent variables.  Check .describe() on the dataset and pay close attention to min/max values comparatively.  
```


```python
def conf_matrix_plot(model, x_data, y_data, normalize=None, title=None):
    """
    Plot confusion matrix for model predictions.
    """

    model_pred = model.predict(x_data)

    cm = confusion_matrix(
        y_data,
        model_pred,
        labels=model.classes_,
        normalize=normalize
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_
    )

    disp.plot(values_format='d' if normalize is None else '.2f')

    if title:
        disp.ax_.set_title(title)

    plt.show()
```

#### Multinomial Naive Bayes


```python
from sklearn.naive_bayes import MultinomialNB
```


```python
# Define X and y
X = data.drop(columns=['target'])
y = data['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=42
)
```


```python

# Fit model
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

# Predictions
y_preds = mnb.predict(X_test)
y_probs = mnb.predict_proba(X_test)[:, 1]
```


```python
# Evaluation
print("Accuracy:", accuracy_score(y_test, y_preds))
print("Precision:", precision_score(y_test, y_preds))
print("Recall:", recall_score(y_test, y_preds))
print("F1:", f1_score(y_test, y_preds))

print("\nDetailed Report:")
print(classification_report(y_test, y_preds))

print("Baseline:", y_test.value_counts(normalize=True).max())
print("ROC AUC:", roc_auc_score(y_test, y_probs))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_preds))

```

#### Bernoulli Naive Bayes


```python
from sklearn.naive_bayes import BernoulliNB

# Define X and y
X = data.drop(columns=['target'])
y = data['target']

# Optional binarization (common)
X = (X > 0).astype(int)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

# Fit model
bnb = BernoulliNB()
bnb.fit(X_train, y_train)

# Predictions
y_preds = bnb.predict(X_test)
y_probs = bnb.predict_proba(X_test)[:, 1]

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_preds))
print("Precision:", precision_score(y_test, y_preds))
print("Recall:", recall_score(y_test, y_preds))
print("F1:", f1_score(y_test, y_preds))

print("\nDetailed Report:")
print(classification_report(y_test, y_preds))

print("Baseline:", y_test.value_counts(normalize=True).max())
print("ROC AUC:", roc_auc_score(y_test, y_probs))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_preds))

```

#### Categorical Naive Bayes


```python
from sklearn.naive_bayes import CategoricalNB

# Define X and y
X = data.drop(columns=['target'])
y = data['target']

# Encode categorical predictors
encoder = OrdinalEncoder()
X = encoder.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

# Fit model
cnb = CategoricalNB()
cnb.fit(X_train, y_train)

# Predictions
y_preds = cnb.predict(X_test)
y_probs = cnb.predict_proba(X_test)[:, 1]

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_preds))
print("Precision:", precision_score(y_test, y_preds))
print("Recall:", recall_score(y_test, y_preds))
print("F1:", f1_score(y_test, y_preds))

print("\nDetailed Report:")
print(classification_report(y_test, y_preds))

print("Baseline:", y_test.value_counts(normalize=True).max())
print("ROC AUC:", roc_auc_score(y_test, y_probs))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_preds))

```

#### Function Module 


```python

def run_nb(model):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]
    print(classification_report(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, probs))

```


```python
run_nb(GaussianNB())
run_nb(MultinomialNB())
run_nb(BernoulliNB())
run_nb(CategoricalNB())

```
