# Chi Squared Test

Determines if two categorical variables are associated with one another and whether a categorical variable follows an expected distribution.

Process:

Identify Null/Alt hypothesis



Calculate the chi-square test statistic



calculate p-value



make conclusion. 


```python
import scipy.stats as stats
import numpy as np
```

### Chi-Square Goodness of Fit Test


```python
observations = []
expectations = []
result = stats.chisquare(f_obs=observations, f_exp=expectations)
result
```

### Chi-Square Test for Independence


```python
observations = np.array([],[],etc)
result = stats.contingency.chi2_contingency(observations, correction=False)
result
```
