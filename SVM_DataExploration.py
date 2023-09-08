#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 18:09:00 2023

@author: jing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
abspath = os.path.abspath(__file__) #current file path
dname = os.path.dirname(abspath)#the directory of this file, dataset is also here
os.chdir(dname)


#%%
diabetes_dataset = pd.read_csv('diabetes.csv')

#diabetes_dataset.head()
#diabetes_dataset.tail()

#765*9
#preg, glucose, BP, skinThick, Insulin, BMI, DBpedigeeFunc, Age, Outcome

describe = diabetes_dataset.describe()

#select by label
bmi = diabetes_dataset["BMI"]
#diabetes_dataset[0:3]
diabetes_dataset.loc[0]
diabetes_dataset.loc[:,['BMI', 'Outcome']]

#select by position
diabetes_dataset.iloc[0]
diabetes_dataset.iloc[0:3, 0:3]
diabetes_dataset.iloc[0:3, :]

#boolean
true_db = diabetes_dataset["Outcome"] == 1
diabetes_dataset[diabetes_dataset['Outcome'] == 1]

#.isin() for filter
BMI_AGE = diabetes_dataset.iloc[:, 5:8]
#BMI_AGE = np.transpose(BMI_AGE)

#%%
#sns.set()
sns.pairplot(BMI_AGE)


