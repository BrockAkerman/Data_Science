# Analysis of Variance (ANOVA)

A group of statistical techniques that test the difference of means between three or more groups.  It is an extension of the t-test (which tests the means of two groups).

**One-way ANOVA**:  Compares the means of one continuous dependent variable based on three or more groups of <u>**one**</u> categorical variable

**Two-way ANOVA**: Compares the means of one continuous dependent variable based on three or more groups of <u>**two**</u> categorical variables  
  
  
There are five steps in performing a one-way ANOVA test:  
1 Calculate group means and grand (overall) mean  
2 Calculate the sum of squares between groups (SSB) and the sum of squares within groups (SSW)  
3 Calculate mean squares for both SSB and SSW  
4 Compute the F-statistic  
5 Use the F-distribution and the F-statistic to get a p-value, which you use to decide whether to reject the null hypothesis 

<u>**Technical note**</u>: The type of an ANOVA and the number of ways of an ANOVA are two distinct concepts: "type" (typ in statsmodels.stats.anova.anova_lm()) refers to how the sums of squares (these quantities are the building blocks for ANOVA) are calculated, while "K-way" means that there are K categorical factors in the analysis.

#### Type 1 ANOVA:  
Sequential SS--“How much variance does each variable explain when added in order?”

Computation logic:  
Fit intercept-only model  
Add A → measure improvement  
Add B → measure additional improvement  
Add interaction → measure additional improvement  

Y ~ A + B ≠ Y ~ B + A

When Type I is appropriate:  
Polynomial regression  
Hierarchical models  
Time/order-dependent predictors  

  
#### Type 2 ANOVA:  
Marginal SS--"Does each main effect matter after accounting for the other main effects, ignoring interactions?"  
  
Computational Logic:  
Test A controlling for B  
Test B controlling for A  
Interaction tested last  
  
Many statisticians consider this the default for balanced experimental designs without strong interactions.  
  
  
#### Type 3 ANOVA:  
Partial SS--"Does each term explain variance after accounting for ALL other terms, including interactions?"  
  
<u>NOTE</u> When interactions exists  
Main effects become conditional effects  
Interpretation changes dramatically




### Assumptions of ANOVA
ANOVA will only work if the following assumptions are true:  
  
**1 The dependent values for each group come from normal distributions**  
Note that this assumption does NOT mean that all of the dependent values, taken together, must be normally distributed. Instead, it means that within each group, the dependent values should be normally distributed.  
  
ANOVA is generally robust to violations of normality, especially when sample sizes are large or similar across groups, due to the central limit theorem. However, significant violations can lead to incorrect conclusions.  
  
**2 The variances across groups are equal**  
ANOVA compares means across groups and assumes that the variance around these means is the same for all groups. If the variances are unequal (i.e., heteroscedastic), it could lead to incorrect conclusions  
  
**3 Observations are independent of each other**  
ANOVA assumes that one observation does not influence or predict any other observation. If there is autocorrelation among the observations, the results of the ANOVA test could be biased.



```python
# Import pandas and seaborn packages
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
```


```python
# Create boxplot to show distribution
sns.boxplot(x = "", y = "", data = data)
```

### **One-Way ANOVA**


```python
# Construct simple linear regression model, and fit the model
model = ols(formula = " ~ ", data = data).fit()
model.summary()
```

NULL Hypothesis: There is not difference in...

$H_0:  \mu_1 = \mu_2 = \mu_3 = ...$

ALT Hypothesis:  There is a difference in...

$H_A:  \mu_1 \neq \mu_2 \neq \mu_3 \neq ...$


```python
# Run one-way ANOVA

# Technical note: The type of an ANOVA and the number of ways of an ANOVA are two distinct concepts: "type" (typ in statsmodels.stats.anova.anova_lm()) refers to how the sums of squares (these quantities are the building blocks for ANOVA) are calculated, while "K-way" means that there are K categorical factors in the analysis.




# LINK:  https://www.statsmodels.org/dev/generated/statsmodels.stats.anova.anova_lm.html



sm.stats.anova_lm(model, typ = 2)
```


```python
sm.stats.anova_lm(model, typ = 1)
```


```python
sm.stats.anova_lm(model, typ = 3)
```

Results are a table that contain F-tests metrics.  We are interested particularly the p-value which will determine whether our hypothesis should be rejected or fail to be rejected.  

### **Two-Way ANOVA**


```python
# Construct a multiple linear regression with an interaction term between color and cut
model2 = ols(formula = " ~ C() + C() + C():C()", data = data).fit()
# Get summary statistics
model2.summary()
```


```python
sm.stats.anova_lm(model2, typ = 2)
```


```python
sm.stats.anova_lm(model2, typ = 1)
```


```python
sm.stats.anova_lm(model2, typ = 3)
```

##### ANOVA Post hoc test

Post hoc test: Performs a pairwise comparison between all available groups while controlling for the error rate.

If we run multiple hypothesis tests all with a 95% confidence level, there is an increasing chance of a false positive, or falsely rejecting the null hypothesis. The post hoc test will control for this, and allows us to run many hypothesis tests while remaining confident with the accuracy of the results. Otherwise, be very careful when running multiple hypothesis tests.


```python
# Import Tukey's HSD function
from statsmodels.stats.multicomp import pairwise_tukeyhsd
```


```python
# Run Tukey's HSD post hoc test for one-way ANOVA
tukey_oneway = pairwise_tukeyhsd(endog = data[''], groups = data[""], alpha = 0.05)
```


```python
# Get results (pairwise comparisons)
tukey_oneway.summary()
```

The tables' reject column tells us which null hypothesis we can reject.  If the value is equal to true, reject the null; If it is not, fail to reject the null. 
