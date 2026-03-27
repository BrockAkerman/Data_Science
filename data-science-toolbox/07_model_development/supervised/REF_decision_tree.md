# Decision Trees


```python
import numpy as np
import pandas as pd
import platform
import sklearn
import xgboost as xgb
print('Python version: ', platform.python_version())
print('numpy version: ', np.__version__)
print('pandas version: ', pd.__version__)
print('sklearn version ', sklearn.__version__)
print('xgboost version ', xgb.__version__)
```

    Python version:  3.12.6
    numpy version:  2.3.5
    pandas version:  2.3.3
    sklearn version  1.8.0
    xgboost version  3.2.0
    


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# This function displays the splits of the tree
from sklearn.tree import plot_tree

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
```


```python
# Read in data
file = 'Churn_Modelling.csv'
df_original = pd.read_csv(file)
df_original.head()
```

    Collecting xgboost
      Downloading xgboost-3.2.0-py3-none-win_amd64.whl.metadata (2.1 kB)
    Requirement already satisfied: numpy in a:\programs\python\lib\site-packages (from xgboost) (2.3.5)
    Requirement already satisfied: scipy in a:\programs\python\lib\site-packages (from xgboost) (1.16.3)
    Downloading xgboost-3.2.0-py3-none-win_amd64.whl (101.7 MB)
       ---------------------------------------- 0.0/101.7 MB ? eta -:--:--
       -- ------------------------------------- 5.8/101.7 MB 29.4 MB/s eta 0:00:04
       --- ------------------------------------ 9.4/101.7 MB 24.5 MB/s eta 0:00:04
       --- ------------------------------------ 9.4/101.7 MB 24.5 MB/s eta 0:00:04
       --- ------------------------------------ 9.7/101.7 MB 15.1 MB/s eta 0:00:07
       --- ------------------------------------ 10.0/101.7 MB 10.7 MB/s eta 0:00:09
       ---- ----------------------------------- 10.5/101.7 MB 8.7 MB/s eta 0:00:11
       ---- ----------------------------------- 11.3/101.7 MB 8.1 MB/s eta 0:00:12
       ---- ----------------------------------- 11.8/101.7 MB 7.3 MB/s eta 0:00:13
       ---- ----------------------------------- 12.6/101.7 MB 6.8 MB/s eta 0:00:14
       ----- ---------------------------------- 13.6/101.7 MB 6.6 MB/s eta 0:00:14
       ----- ---------------------------------- 15.2/101.7 MB 6.6 MB/s eta 0:00:14
       ------ --------------------------------- 17.0/101.7 MB 6.9 MB/s eta 0:00:13
       ------- -------------------------------- 18.4/101.7 MB 6.8 MB/s eta 0:00:13
       ------- -------------------------------- 20.2/101.7 MB 7.0 MB/s eta 0:00:12
       -------- ------------------------------- 21.0/101.7 MB 6.8 MB/s eta 0:00:12
       -------- ------------------------------- 22.3/101.7 MB 6.7 MB/s eta 0:00:12
       --------- ------------------------------ 23.9/101.7 MB 6.8 MB/s eta 0:00:12
       ---------- ----------------------------- 26.0/101.7 MB 7.0 MB/s eta 0:00:11
       ----------- ---------------------------- 28.0/101.7 MB 7.2 MB/s eta 0:00:11
       ----------- ---------------------------- 30.1/101.7 MB 7.3 MB/s eta 0:00:10
       ------------ --------------------------- 31.7/101.7 MB 7.3 MB/s eta 0:00:10
       ------------ --------------------------- 32.8/101.7 MB 7.2 MB/s eta 0:00:10
       ------------- -------------------------- 34.9/101.7 MB 7.4 MB/s eta 0:00:10
       -------------- ------------------------- 37.5/101.7 MB 7.6 MB/s eta 0:00:09
       --------------- ------------------------ 39.8/101.7 MB 7.7 MB/s eta 0:00:08
       ---------------- ----------------------- 41.7/101.7 MB 7.8 MB/s eta 0:00:08
       ----------------- ---------------------- 43.5/101.7 MB 7.8 MB/s eta 0:00:08
       ----------------- ---------------------- 45.1/101.7 MB 7.8 MB/s eta 0:00:08
       ------------------ --------------------- 46.9/101.7 MB 7.9 MB/s eta 0:00:07
       ------------------- -------------------- 49.5/101.7 MB 8.0 MB/s eta 0:00:07
       -------------------- ------------------- 51.6/101.7 MB 8.1 MB/s eta 0:00:07
       -------------------- ------------------- 53.2/101.7 MB 8.1 MB/s eta 0:00:06
       --------------------- ------------------ 55.1/101.7 MB 8.1 MB/s eta 0:00:06
       ---------------------- ----------------- 57.1/101.7 MB 8.2 MB/s eta 0:00:06
       ----------------------- ---------------- 59.0/101.7 MB 8.2 MB/s eta 0:00:06
       ----------------------- ---------------- 60.6/101.7 MB 8.2 MB/s eta 0:00:06
       ------------------------ --------------- 61.6/101.7 MB 8.1 MB/s eta 0:00:05
       ------------------------ --------------- 63.2/101.7 MB 8.1 MB/s eta 0:00:05
       ------------------------- -------------- 65.3/101.7 MB 8.1 MB/s eta 0:00:05
       -------------------------- ------------- 66.6/101.7 MB 8.1 MB/s eta 0:00:05
       -------------------------- ------------- 68.4/101.7 MB 8.1 MB/s eta 0:00:05
       --------------------------- ------------ 70.3/101.7 MB 8.2 MB/s eta 0:00:04
       --------------------------- ------------ 70.8/101.7 MB 8.1 MB/s eta 0:00:04
       ---------------------------- ----------- 71.8/101.7 MB 8.0 MB/s eta 0:00:04
       ---------------------------- ----------- 73.4/101.7 MB 7.9 MB/s eta 0:00:04
       ----------------------------- ---------- 74.7/101.7 MB 7.9 MB/s eta 0:00:04
       ----------------------------- ---------- 75.5/101.7 MB 7.8 MB/s eta 0:00:04
       ------------------------------ --------- 76.8/101.7 MB 7.8 MB/s eta 0:00:04
       ------------------------------ --------- 77.9/101.7 MB 7.7 MB/s eta 0:00:04
       ------------------------------- -------- 79.4/101.7 MB 7.7 MB/s eta 0:00:03
       ------------------------------- -------- 81.3/101.7 MB 7.8 MB/s eta 0:00:03
       -------------------------------- ------- 82.1/101.7 MB 7.7 MB/s eta 0:00:03
       -------------------------------- ------- 83.1/101.7 MB 7.6 MB/s eta 0:00:03
       -------------------------------- ------- 83.6/101.7 MB 7.6 MB/s eta 0:00:03
       --------------------------------- ------ 84.4/101.7 MB 7.5 MB/s eta 0:00:03
       --------------------------------- ------ 85.5/101.7 MB 7.4 MB/s eta 0:00:03
       ---------------------------------- ----- 86.8/101.7 MB 7.4 MB/s eta 0:00:03
       ---------------------------------- ----- 87.8/101.7 MB 7.4 MB/s eta 0:00:02
       ----------------------------------- ---- 89.1/101.7 MB 7.4 MB/s eta 0:00:02
       ----------------------------------- ---- 89.9/101.7 MB 7.3 MB/s eta 0:00:02
       ----------------------------------- ---- 91.0/101.7 MB 7.3 MB/s eta 0:00:02
       ------------------------------------ --- 92.8/101.7 MB 7.3 MB/s eta 0:00:02
       ------------------------------------- -- 94.9/101.7 MB 7.3 MB/s eta 0:00:01
       ------------------------------------- -- 95.9/101.7 MB 7.3 MB/s eta 0:00:01
       -------------------------------------- - 97.5/101.7 MB 7.3 MB/s eta 0:00:01
       ---------------------------------------  99.6/101.7 MB 7.4 MB/s eta 0:00:01
       ---------------------------------------  100.7/101.7 MB 7.3 MB/s eta 0:00:01
       ---------------------------------------  101.4/101.7 MB 7.3 MB/s eta 0:00:01
       ---------------------------------------  101.4/101.7 MB 7.3 MB/s eta 0:00:01
       ---------------------------------------- 101.7/101.7 MB 7.1 MB/s eta 0:00:00
    Installing collected packages: xgboost
    Successfully installed xgboost-3.2.0
    

    
    [notice] A new release of pip is available: 24.2 -> 26.0.1
    [notice] To update, run: python.exe -m pip install --upgrade pip
    


```python
# Check class balance
df_original['Exited'].value_counts()
```


```python
# Calculate average balance of customers who churned
avg_churned_bal = df_original[df_original['Exited']==1]['Balance'].mean()
avg_churned_bal
```


```python
# Create a new df that drops RowNumber, CustomerId, Surname, and Gender cols
churn_df = df_original.drop(['RowNumber', 'CustomerId', 'Surname', 'Gender'], 
                            axis=1)

churn_df.head()
```


```python
# Dummy encode categorical variables
churn_df = pd.get_dummies(churn_df, drop_first=True)
churn_df.head()
```


```python
# Define the y (target) variable
y = churn_df['Exited']

# Define the X (predictor) variables
X = churn_df.copy()
X = X.drop('Exited', axis=1)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25, stratify=y, 
                                                    random_state=42)
```


```python
# Instantiate the model
decision_tree = DecisionTreeClassifier(random_state=0)

# Fit the model to training data
decision_tree.fit(X_train, y_train)

# Make predictions on test data
dt_pred = decision_tree.predict(X_test)
```


```python
# Generate performance metrics
print("Accuracy:", "%.3f" % accuracy_score(y_test, dt_pred))
print("Precision:", "%.3f" % precision_score(y_test, dt_pred))
print("Recall:", "%.3f" % recall_score(y_test, dt_pred))
print("F1 Score:", "%.3f" % f1_score(y_test, dt_pred))
```


```python
def conf_matrix_plot(model, x_data, y_data):
    '''
    Accepts as argument model object, X data (test or validate), and y data (test or validate). 
    Returns a plot of confusion matrix for predictions on y data.
    ''' 
  
    model_pred = model.predict(x_data)
    cm = confusion_matrix(y_data, model_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=model.classes_)
  
    disp.plot(values_format='')  # `values_format=''` suppresses scientific notation
    plt.show()
```


```python
# Generate confusion matrix
conf_matrix_plot(decision_tree, X_test, y_test)
```


```python
# Plot the tree
plt.figure(figsize=(15,12))
plot_tree(decision_tree, max_depth=2, fontsize=14, feature_names=X.columns, 
          class_names={0:'stayed', 1:'churned'}, filled=True);
plt.show()
```


```python
# Import GridSearchCV
from sklearn.model_selection import GridSearchCV
```


```python
# Assign a dictionary of hyperparameters to search over
tree_para = {'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50],
             'min_samples_leaf': [2, 5, 10, 20, 50]}
```


```python
# Assign a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1'}
```


```python
# Instantiate the classifier
tuned_decision_tree = DecisionTreeClassifier(random_state = 42)
```


```python
# Instantiate the GridSearch
clf = GridSearchCV(tuned_decision_tree, 
                   tree_para, 
                   scoring = scoring, 
                   cv=5, 
                   refit="f1")

# Fit the model
clf.fit(X_train, y_train)
```


```python
# Examine the best model from GridSearch
clf.best_estimator_
```


```python
print("Best Avg. Validation Score: ", "%.4f" % clf.best_score_)
```


```python
def make_results(model_name, model_object):
    '''
    Accepts as arguments a model name (your choice - string) and
    a fit GridSearchCV model object.
  
    Returns a pandas df with the F1, recall, precision, and accuracy scores
    for the model with the best mean F1 score across all validation folds.  
    '''

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(mean f1 score)
    best_estimator_results = cv_results.iloc[cv_results['mean_test_f1'].idxmax(), :]

    # Extract accuracy, precision, recall, and f1 score from that row
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy
  
    # Create table of results
    table = pd.DataFrame({'Model': [model_name],
                        'F1': [f1],
                        'Recall': [recall],
                        'Precision': [precision],
                        'Accuracy': [accuracy]
                         }
                        )
  
    return table
```


```python
# Call the function on our model
result_table = make_results("Tuned Decision Tree", clf)
```


```python
# Save results table as csv
result_table.to_csv("Results.csv")
```


```python
# View the results
result_table
```


```python

```


```python

```


```python

```


```python

```
