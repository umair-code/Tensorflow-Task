# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 21:47:13 2019

@author: Umair Riaz
"""
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
Machine Learning - Lab-6
"""
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
Task-1
Decision Tree Classifier
Car Evaluation Dataset - http://archive.ics.uci.edu/ml/datasets/Car+Evaluation
"""
#'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree, metrics
#import sklearn.cross_validation
from sklearn.model_selection import train_test_split

data = pd.read_csv('car.data',names=['buying','maint','doors','persons','lug_boot','safety','class'])

data.head()

data.info()

data['class'],class_names = pd.factorize(data['class'])

print(class_names)
print(data['class'].unique())

data['buying'],_ = pd.factorize(data['buying'])
data['maint'],_ = pd.factorize(data['maint'])
data['doors'],_ = pd.factorize(data['doors'])
data['persons'],_ = pd.factorize(data['persons'])
data['lug_boot'],_ = pd.factorize(data['lug_boot'])
data['safety'],_ = pd.factorize(data['safety'])
data.head()

data.info()

X = data.iloc[:,:-1]
y = data.iloc[:,-1]

# split data randomly into 70% training and 30% test
#X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# train the decision tree
dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
dtree.fit(X_train, y_train)

# use the model to make predictions with the test data
y_pred = dtree.predict(X_test)

# how did our model perform?
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))
#'''
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
Task-2
ExtraTreesClassifier - An extremely randomized tree classifier.

This class implements a meta estimator that fits a number of
randomized decision trees (a.k.a. extra-trees) on various 
sub-samples of the dataset and uses averaging to improve 
the predictive accuracy and control over-fitting.
"""
'''
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')

# Build a classification task using 3 informative features
X, y = make_classification(n_samples=1000,
                           n_features=10,
                           n_informative=3,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=False)

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
'''
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
Task-3
Decision Tree Classifier
Analyzing Classification Accuracy
"""
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import graphviz

df = pd.read_csv('titanic1.csv', index_col='PassengerId')
print(df.head(3))

df = df[['Pclass', 'Gender', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
print(df)

#df = df[['Gender', 'Age']]
#print(df)

df['Gender']=df['Gender'].map({'male':0,'female':1})
print(df)

#print(df1.shape)
### Drop any rows with missing values.
df = df.dropna()
print(df.shape)

X = df.drop('Survived', axis=1)
y = df['Survived']

print(X.shape)
print(y.shape)

### Training
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

### Decision Tree instance
model = tree.DecisionTreeClassifier()

### Model’s attributes
print('Decision Tree Model: ',model)

### Model training
model.fit(X_train, y_train)

###score the predicted output from model on our test data against our ground truth test data
y_predict = model.predict(X_test)
accuracy_score(y_test, y_predict)

### Analyzing confusion matrix
from sklearn.metrics import confusion_matrix

pd.DataFrame(
    confusion_matrix(y_test, y_predict),
    columns=['Predicted Not Survival', 'Predicted Survival'],
    index=['True Not Survival', 'True Survival'])
'''
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
Task-4
Feature Importance
"""
'''
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.datasets import load_iris

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

train, test = df[df['is_train']==True], df[df['is_train']==False]
features = df.columns[0:4]

### RANDOM FORESTS MODEL
forest = RFC(n_jobs=2,n_estimators=50)

y, _ = pd.factorize(train['species'])

### Result analysis
forest.fit(train[features], y)
preds = iris.target_names[forest.predict(test[features])]
print(pd.crosstab(index=test['species'], columns=preds, rownames=['actual'], colnames=['preds']))
'''
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
Task-6
Blobs
"""
'''
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
from pandas import DataFrame
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')

# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=3, n_features=2)

# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue', 2:'green'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()
''' 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







