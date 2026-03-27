# Multiple Linear Regression (MLR)

##### Overview

Multiple Linear Regression (MLR) models the relationship between a continuous response variable $Y$ and multiple predictor variables $X_1, X_2, \dots, X_p$.

$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p + \varepsilon
$$

Where:

- $\beta_0$ = intercept  
- $\beta_j$ = expected change in $Y$ for a one-unit increase in $X_j$, holding other predictors constant  
- $\varepsilon$ = random error term  

---

##### When to Use MLR

Use Multiple Linear Regression when:

- The response variable is **continuous**
- Relationships between predictors and response are approximately **linear**
- You need:
  - Effect size estimation
  - Statistical inference (p-values, confidence intervals)
  - Prediction
  - Control for confounding variables
- Interpretability is important

---

##### Model Assumptions

1. **Linearity**  
   The relationship between predictors and the response is linear.

2. **Independence**  
   Observations are independent of one another.

3. **Homoscedasticity**  
   The variance of residuals is constant across fitted values:
   
   $$
   \text{Var}(\varepsilon_i \mid X) = \sigma^2
   $$

4. **Normality of Errors**  
   Residuals are approximately normally distributed:
   
   $$
   \varepsilon_i \sim \mathcal{N}(0, \sigma^2)
   $$

5. **No Perfect Multicollinearity**  
   Predictors are not perfectly linearly related.

---

##### Diagnostics to Evaluate Assumptions

- Residual vs Fitted Plot → Linearity & Homoscedasticity  
- Q-Q Plot → Normality  
- Variance Inflation Factor (VIF) → Multicollinearity  
- Cook’s Distance → Influential observations  
- Condition Number → Numerical stability  

---

##### Strengths

- Highly interpretable coefficients  
- Strong inferential framework  
- Computationally efficient  
- Well-understood theoretical properties  

---

##### Limitations

- Sensitive to outliers  
- Assumes linearity  
- Performance degrades under strong multicollinearity  
- Can overfit with many predictors  
- Sensitive to heteroscedasticity unless corrected  

---

##### Interpretation Notes

- Coefficients represent **partial effects** (holding other variables constant)
- $R^2$ measures variance explained
- Adjusted $R^2$ penalizes additional predictors
- Statistical significance does not imply practical significance

---

##### Common Extensions

- Interaction terms  
- Polynomial regression  
- Robust standard errors  
- Ridge / Lasso regression  
- Generalized Linear Models (GLMs)  


## Performing Multiple Linear Regression

Preliminary steps include import revelant libraries and modules, setting the working directory, importing the dataset, and showcasing the summary statistics of the dataset to capture 


```python
#Import Libraries and Modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os

#Set Working Directory
os.chdir(r"A:\Personal Files\Career Folder\Data_Science\data-science-toolbox")
```


```python
#Load and inspect data
data = pd.read_csv("datasets/examples/creditcard.csv")
```


```python
print(data.head())
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



#### Data Preprocessing


```python
#Clean Data
#data = data.dropna()
```


```python
#Encoding
#data = pd.get_dummies(data, drop_first=True)
```


```python
#Fix formatting for MLR.  
#Ols does not accept labels with spaces

data.columns = data.columns.str.strip().str.replace(r"\s+", "", regex=True)

```

#### Define X and Y


```python
y = data['']
x = data[['','']]
X = sm.add_constant(X)
```


```python
model = sm.OLS(y,X).fit()
```


```python
print(model.summary())
```

#### Assumptions Check


```python
plt.scatter(model.fittedvalues, model.resid)
plt.axhline(0)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.show()
```


```python

sns.histplot(model.resid, kde=True)

```


```python
#Homoscedasticity
from statsmodels.stats.diagnostic import het_breuschpagan

het_breuschpagan(model.resid, model.model.exog)

```


```python
#Multicollinearity
#  VIF > 5  == concernings
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i)
              for i in range(X.shape[1])]

print(vif)

```


```python

```


```python

```


```python

```


```python

```


```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
```


