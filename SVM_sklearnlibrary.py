#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 21:19:14 2023

@author: jing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

import plotly.express as ex
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import ROCAUC
from yellowbrick.style import set_palette

import sklearn
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn import svm

import os
abspath = os.path.abspath(__file__) #current file path
dname = os.path.dirname(abspath)#the directory of this file, dataset is also here
os.chdir(dname)
#%%
#Inspect dataframe structure
diabetes_dataset = pd.read_csv('diabetes.csv')
df = diabetes_dataset.copy()



# Check first 5 lines
df.head()

# Check last 5 lines
df.tail()

# Check Dataframe information
df.info()

# Null Data
df.isna().sum()

# Numerical features
df.describe(exclude=['O'])

# Check row and column numbers
rows = df.shape[0] 
cols = df.shape[1] 
print("Rows   : " + str(rows)) 
print("Columns: " + str(cols))

# Check duplicate data
print("Number of duplicates: " + str(df.duplicated().sum()))

# Number of unique values per column.
df.nunique()

# Check the unique values and frequency for 'Outcome'
df['Outcome'].value_counts()

#%%
#exploratory data analysis
# Outcome - Target Feature

fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(x=df["Outcome"], palette="magma")
#plt.show()

# Age

fig, ax = plt.subplots(figsize=(18, 4))
sns.countplot(x=df["Age"], palette="tab10")
#plt.show()

# Pregnancies

fig, ax = plt.subplots(figsize=(18, 4))
sns.countplot(x=df["Pregnancies"], palette="tab10")
#plt.show()

# Histogram Dataset

df.hist(bins=50, figsize=(12,12))
#plt.show()


# Check data types in df
numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']
discrete_features = [feature for feature in numerical_features if len(df[feature].unique())<25]
continuous_features = [feature for feature in numerical_features if feature not in discrete_features]
categorical_features = [feature for feature in df.columns if feature not in numerical_features]
binary_categorical_features = [feature for feature in categorical_features if len(df[feature].unique()) <=3]
print("Numerical Features Count {}".format(len(numerical_features)))
print("Discrete features Count {}".format(len(discrete_features)))
print("Continuous features Count {}".format(len(continuous_features)))
print("Categorical features Count {}".format(len(categorical_features)))
print("Binary Categorical features Count {}".format(len(binary_categorical_features)))

#%%
# Distribution

def generate_distribution_plot(train_df, continuous_features):
    # create copy of dataframe
    data = train_df[continuous_features].copy()
    # Create subplots 
    fig, axes = plt.subplots(nrows=len(data.columns)//3, ncols=4,figsize=(15,10))
    fig.subplots_adjust(hspace=0.7)
    
    # set fontdict
    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
    
    # Generate distplot
    for ax, feature in zip(axes.flatten(), data.columns):
        feature_mean = data[feature].mean()
        feature_median = data[feature].median()
        feature_mode = data[feature].mode().values[0]
        sns.distplot(data[feature],ax=ax)
        ax.set_title(f'Analysis of {feature}', fontdict=font)
        ax.axvline(feature_mean, color='r', linestyle='--', label="Mean")
        ax.axvline(feature_median, color='g', linestyle='--', label="Median")
        ax.axvline(feature_mode, color='b', linestyle='--', label="Mode")
        ax.legend()
    plt.show()
    
# Print Distribution
generate_distribution_plot(df, continuous_features)

# Correlations

limit = -1.0

data = df.corr()["Outcome"].sort_values(ascending=False)
indices = data.index
labels = []
corr = []
for i in range(1, len(indices)):
    if data[indices[i]]>limit:
        labels.append(indices[i])
        corr.append(data[i])
sns.barplot(x=corr, y=labels)
plt.title('Correlations with "Outcome"')
plt.show()

#Super SNS funciton: relationship plot
sns.relplot(data=df, x="Glucose", y="BMI", hue="Outcome")
plt.show()

# Heatmap

df_corr = df.corr()
f, ax = plt.subplots(figsize=(6, 6))

sns.heatmap(df_corr, annot=True, fmt='.2f', cmap='RdYlGn',annot_kws={'size': 8}, ax=ax)
plt.show()

#%%
#so many algorithms wth
# assign x and y values
x,y=df.drop("Outcome",axis=1),df[['Outcome']]

from sklearn.model_selection import train_test_split
# split the data to train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
x_train.shape,x_test.shape,y_train.shape,y_test.shape



# Classification Algorithms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

g=GaussianNB()
b=BernoulliNB()
k=KNeighborsClassifier()
l=LogisticRegression()
d=DecisionTreeClassifier()
r=RandomForestClassifier()
gb=GradientBoostingClassifier()

# fit and predict model
g.fit(x,y)
b.fit(x,y)
k.fit(x,y)
l.fit(x,y)
d.fit(x,y)
r.fit(x,y)
gb.fit(x,y)

predG=g.predict(x)
predB=b.predict(x)
predK=k.predict(x)
predL=l.predict(x)
predD=d.predict(x)
predR=r.predict(x)
predGB=gb.predict(x)

# Print 'accuracy scores'Å
print('Accuracy Scores:')
print("GaussianNB:       ", accuracy_score(predG,y))
print("BernoulliNB:      ", accuracy_score(predB,y))
print("KNeighbours:      ", accuracy_score(predK,y))
print("LogisticReg:      ", accuracy_score(predL,y))
print("DecisionTree:     ", accuracy_score(predD,y))
print("RandomForest:     ", accuracy_score(predR,y))
print("GradientBoosting: ", accuracy_score(predGB,y))
#%%




















#%%
#replace unreasonble values
diabetes_dataset = pd.read_csv('diabetes.csv')

category_cannotbezero = diabetes_dataset.loc[:, ['Glucose','BloodPressure', 'SkinThickness']]

for category in category_cannotbezero:
    diabetes_dataset[category] = diabetes_dataset[category].replace(0, np.NAN)
    mean = diabetes_dataset[category].mean(skipna = True)
    diabetes_dataset[category] = diabetes_dataset[category].replace(np.NAN, mean)
    

df = diabetes_dataset

#%%
# Splitting the dataset into training and testing sets.
x = df.iloc[:, :-2]
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, test_size = 0.2)

#%%
# Creating the SVM model.
clf = svm.SVC(kernel='rbf')
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

#%%
print("Accuracy:", accuracy_score(y_test, y_pred))
#0.7922

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)
#array([[98,  9],
#       [23, 24]])

