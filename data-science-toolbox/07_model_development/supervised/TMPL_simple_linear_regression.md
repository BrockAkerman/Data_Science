# Simple Linear Regression (SLR)

## Overview

Simple Linear Regression (SLR) models the relationship between a continuous response variable $Y$ and a single predictor variable $X$.

$$
Y = \beta_0 + \beta_1 X + \varepsilon
$$

Where:

- $\beta_0$ = intercept (expected value of $Y$ when $X = 0$)
- $\beta_1$ = expected change in $Y$ for a one-unit increase in $X$
- $\varepsilon$ = random error term

---

## When to Use SLR

Use Simple Linear Regression when:

- The response variable is **continuous**
- The relationship between the predictor and response is approximately **linear**
- You need:
  - Effect size estimation
  - Statistical inference (p-values, confidence intervals)
  - Prediction
- Interpretability is important
- Only **one primary explanatory variable** is being studied

---

When NOT to Use SLR:  
- nonlinear pattern  
- heteroscedastic funnel shape  
- strong outliers dominating fit  

## Model Assumptions

### 1. Linearity
The relationship between $X$ and the response is linear.

### 2. Independence
Observations are independent of one another.

### 3. Homoscedasticity
The variance of residuals is constant across fitted values:

$$
\mathrm{Var}(\varepsilon_i \mid X_i) = \sigma^2
$$

### 4. Normality of Errors
Residuals are approximately normally distributed:

$$
\varepsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

---

#### Diagnostics to Evaluate Assumptions

- Residual vs Fitted Plot → Linearity & Homoscedasticity
- Q-Q Plot → Normality of residuals
- Residuals vs Predictor Plot → Functional form assessment
- Cook’s Distance → Influential observations

---

#### Strengths

- Extremely interpretable coefficients
- Clear visualization of model fit
- Strong inferential framework
- Computationally simple and efficient
- Serves as a foundation for more complex regression models

---

#### Limitations

- Sensitive to outliers and influential points
- Assumes a linear relationship
- Cannot account for additional explanatory variables
- Sensitive to heteroscedasticity unless corrected

---

#### Interpretation Notes

- $\beta_1$ represents the **expected change in the mean of $Y$** for a one-unit increase in $X$
- $R^2$ measures the proportion of variance in $Y$ explained by $X$
- Statistical significance does not imply practical significance
- Extrapolation beyond the observed range of $X$ may be unreliable

---

#### Common Extensions

- Polynomial regression
- Transformation of variables (log, square root, etc.)
- Robust standard errors
- Weighted least squares
- Generalized Linear Models (GLMs)

---

#### SLR Workflow Checklist

1. Inspect data
2. Visualize X vs Y
3. Confirm approximate linearity
4. Fit OLS model
5. Check residual assumptions
6. Interpret coefficients
7. Generate predictions

---

The following sections implement a complete SLR workflow from raw data to validated model.

## Performing Simple Linear Regression

### Environment Setup

Preliminary steps include import revelant libraries and modules, setting the working directory, importing the dataset, and showcasing the summary statistics of the dataset to capture 


```python
# Import Libraries and Modules

## 1. Setup and Data loading
import os
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import het_breuschpagan

# Set Working Directory
os.chdir(r"A:\Personal Files\Career Folder\Data_Science\data-science-toolbox")

# Load Data
data = pd.read_csv("datasets/examples/creditcard.csv")
```

### Data Inspection


```python
## 2. Initial Data Understanding
print(data.head(5))
```


```python
print(data.info())
```


```python
print(data.describe())
```


```python
#Account for missing values.  
na_counts = data.isna().sum()
na_percent = (na_counts / len(data)) * 100

missing_summary = pd.DataFrame({
    "Missing Count": na_counts,
    "Missing Percent": na_percent
})

missing_summary[missing_summary["Missing Count"] > 0]

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Missing Count</th>
      <th>Missing Percent</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



### Exploratory Data Analysis


```python
## 3. Exploratory Analysis
# MODEL ASSUMPTION #1:  Linearity relationship between independent and dependent variable.
plt.scatter(data[X], data[Y])
```

### Data Preparation


```python
## 4. Data Prepartion
## Clean Data
data = data.dropna()
```

### Define Model Variables


```python
y = data['']
x = data[['','']]
X = sm.add_constant(X) #Statsmodels does not automatically include an intercept term, so we explicitly add a constant column.
```

### Fit the Model


```python
## 5. Fit the SLR Model
model = sm.OLS(y,X).fit()
```


```python
print(model.summary())
```

### Model Diagnostics


```python
## 6. Model Diagnostics
# Fitted vs Residual plotting
plt.scatter(model.fittedvalues, model.resid)
plt.axhline(0)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.show()
```


```python
# A second fitted vs residual plotting
plt.scatter(X, residuals)
plt.axhline(0)
```


```python
# MODEL ASSUMPTION #3: Homoscedasticity--variance is appromally random and even across X.
het_breuschpagan(model.resid, model.model.exog)
```


```python

# MODEL ASSUMPTION #4: Normality of Residual Error--residuals are approximately normallly distributed
sns.histplot(model.resid, kde=True)

```

### Model Interpretation

Slope Interpretation:  
Intercept Meaning:  
P-Value Interpretation:  
R-sq meaning:  

### Prediction & Usage


```python
## 7. Prediction
model.predict(new_X)
```

### Model Improvement

Can optionally include tranformationas and polynomial terms
