# Logistic Regression

A technique that models a categorical dependent variable (Y) based on one or more independent variables (X).

## Binomial Logistic Regression

A technique that models the probability of an observation falling into one of two categories, based on one or more independent variables.

Assumptions

Linearity:  There should be a linear relationship between each X variable and the logit of the probability that Y equals 1. 

Independent observations:

No multicollinearity

No extreme outliers

---





logit odds = p/1-p

logit (long-odds) = log(p/1-p)
The logarithm of the odds of a given probability.  So the logit of probability p is equal to the logarithm of p divided by 1 minus p. 

Maximum likelihood estimation (MLE) 
a technique for estimating the beta parameters that maximize the likelihood of the model producing the observed data. 

Likelihood 
the probability of observing the actual data, given some set of beta parameters. 



```python
#Plots true negatives, true positives 

import sklearn.metrics as metrics
cm = metrics.confusion_matrix(y_test, y_pred, labels = clf.classes_)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = clf.classes_)
```



Precision = Proportion of positive predictions that were true positives

Precision = tp/tp+fp

import sklearn.metrics as metrics
metrics.precision_score(y_test, y_pred)


Recall = Proprotion of positives the model was able to ID correctly.

Recall = TP / TP+FN

metrics.recall_score(y_test, y_pred)



Accuracy = Proportion of data points that were correctly categorized

Accuracy = TP+TN/Tpred

metrics.accuracy_score(y_test, y_pred)




Receiver Operating Characterist (ROC) curve

import matplotlib.pyplot as plt

from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_predictions(y_test, y_pred)



Area Under the Curve (AUC) Curve
metrics.roc_auc_score(y_test,y_pred)

Logistic Regression Interpretations

For every one-unit increase in the vertical acceleration, we expect that the odds the person is lying down decreases by 11%



when providing interpretations of the data, it is helpful to add Confusion Matrix, Accuracy, Precision, Recall, ROC/AUC. 

When to use precision
Using precision as an evaluation metric is especially helpful in contexts where the cost of a false positive is quite high and much higher than the cost of a false negative. For example, in the context of email spam detection, a false positive (predicting a non-spam email as spam) would be more costly than a false negative (predicting a spam email as non-spam). A non-spam email that is misclassified could contain important information, such as project status updates from a vendor to a client or assignment deadline announcements from an instructor to a class of students. 

When to use recall
Using recall as an evaluation metric is especially helpful in contexts where the cost of a false negative is quite high and much higher than the cost of a false positive. For example, in the context of fraud detection among credit card transactions, a false negative (predicting a fraudulent credit card charge as non-fraudulent) would be more costly than a false positive (predicting a non-fraudulent credit card charge as fraudulent). A fraudulent credit card charge that is misclassified could lead to the customer losing money, undetected.

When to use accuracy
It is helpful to use accuracy as an evaluation metric when you specifically want to know how much of the data at hand has been correctly categorized by the classifier. Another scenario to consider: accuracy is an appropriate metric to use when the data is balanced, in other words, when the data has a roughly equal number of positive examples and negative examples. Otherwise, accuracy can be biased. For example, imagine that 95% of a dataset contains positive examples, and the remaining 5% contains negative examples. Then you train a logistic regression classifier on this data and use this classifier predict on this data. If you get an accuracy of 95%, that does not necessarily indicate that this classifier is effective. Since there is a much larger proportion of positive examples than negative examples, the classifier may be biased towards the majority class (positive) and thus the accuracy metric in this context may not be meaningful. When the data you are working with is imbalanced, consider either transforming it to be balanced or using a different evaluation metric other than accuracy. 
