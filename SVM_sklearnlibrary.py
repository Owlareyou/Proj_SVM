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

