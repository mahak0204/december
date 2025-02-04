# -*- coding: utf-8 -*-
"""Practical_7_MLT.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1H6xonlRNqnVYS7b5_GqE_4ofjfbXn9Zz
"""

# https://www.kaggle.com/datasets/elikplim/car-evaluation-data-set

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

df=pd.read_csv("car_evaluation.csv")

df.head()

new_names={'vhigh':'buying', 'vhigh.1':'maint', '2':'doors', '2.1':'persons', 'small':'lug_boot', 'low':'safety', 'unacc':'class'}
df.rename(columns=new_names, inplace=True)
df.head()

df.info()

df['buying'].value_counts(),df['maint'].value_counts(),df['doors'].value_counts(),df['persons'].value_counts(),df['lug_boot'].value_counts(),df['safety'].value_counts()

X=df.drop('class', axis=1)
y=df['class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

!pip install category_encoders

import category_encoders as ce
encoder=ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
X_train=encoder.fit_transform(X_train)
X_test=encoder.transform(X_test)

X_train['buying'].value_counts()

from sklearn.tree import DecisionTreeClassifier

clf_gini=DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=0)
clf_gini.fit(X_train, y_train)

y_pred_gini=clf_gini.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
print(accuracy_score(y_test,y_pred_gini))
print(classification_report(y_test, y_pred_gini))
print(confusion_matrix(y_test, y_pred_gini))

clf_gini.score(X_train,y_train)

clf_gini.score(X_test,y_test)

plt.figure(figsize=(12,8))

from sklearn import tree

tree.plot_tree(clf_gini.fit(X_train, y_train))

