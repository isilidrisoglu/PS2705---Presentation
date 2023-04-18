#!/usr/bin/env python
# coding: utf-8

## Titanic Survival Classification
#Isil Idrisoglu

# imports
import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# check if dataset exist
get_ipython().system('ls')

df = pd.read_csv("./titanic-dataset.csv")
print("Dataset shape: ", df.shape)
df.head()


## Data Preparation
# Check the distribution of Survived
df['Survived'].value_counts()


# Drop irrelevant variables, make sex a dummy variable
df.drop(['PassengerId','Name','Ticket','Fare','Embarked'],axis=1, inplace=True)
df.loc[df['Sex']=='male','Sex']=1
df.loc[df['Sex']=='female','Sex']=0


# Check the percentage of null data
((df.isnull().sum())/len(df))*100

#Drop cabin due to so many missing variables
df.drop('Cabin', axis=1,inplace=True)

#Replace null values in Age
df['Age'].fillna(df['Age'].mean(), inplace = True)

df.info()


## Model
X = df.drop('Survived', axis=1)  # features
y = df['Survived']  # labels

X.shape, y.shape


#Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

X_train.shape, X_test.shape


## Data Standardization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

scaled_X_train= scaler.transform(X_train)
scaled_X_test= scaler.transform(X_test)


## SVM Model
# We are applying soft SVM here, with default C value and linear kernel
from sklearn.svm import SVC

model = SVC(kernel='linear')
model.fit(scaled_X_train, y_train)

y_pred = model.predict(scaled_X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


## Hyperparameter Tuning
#So far we used default values for hyperparameters, which is C.

#This gives us the implementation details and the parameters.
help(SVC)

from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore') # in order to filter some sklearn warnings

svm = SVC(max_iter=500)
param_grid = {'C':[0.01, 0.1, 1, 10], 'gamma': [1, 0.1, 0.01, 0.001],
             'kernel': ['linear', 'rbf']}
grid = GridSearchCV(svm, param_grid)
grid.fit(scaled_X_train, y_train)

grid.best_params_

#Final improved model
y_pred_grid = grid.predict(scaled_X_test)
print(classification_report(y_test, y_pred_grid))

