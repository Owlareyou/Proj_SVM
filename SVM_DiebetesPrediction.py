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
#replace unreasonble values
diabetes_dataset = pd.read_csv('diabetes.csv')

category_cannotbezero = diabetes_dataset.loc[:, ['Glucose','BloodPressure', 'SkinThickness']]

for category in category_cannotbezero:
    diabetes_dataset[category] = diabetes_dataset[category].replace(0, np.NAN)
    mean = diabetes_dataset[category].mean(skipna = True)
    diabetes_dataset[category] = diabetes_dataset[category].replace(np.NAN, mean)
    

diabetes_dataset.describe() #no more zeros in glucose, BP, ST

#%%
#train test split
#test
test = diabetes_dataset.iloc[600:768,:]
#train
diabetes_dataset = diabetes_dataset.iloc[0:600,:]



#%%
all_columns = diabetes_dataset.columns
num_w = all_columns.size -1

diabetes_parameter = diabetes_dataset.iloc[:, 0:8]
diabetes_outcome = diabetes_dataset.iloc[:,8]
diabetes_outcome_withzeros = diabetes_outcome
diabetes_outcome = diabetes_outcome.replace(0,-1)


#initialize w, b
w = np.ones(num_w) #+np.ones(num_w)
b = 0
alpha = 1

#for future accuracy accessiment 
def train_test_split():
    return

#f(x)
def pred_value(w, b, diabetes_parameter):
    pred_value = w*diabetes_parameter +b
    pred_value = pred_value.sum(axis = 1)
    
    #temp fix for bad initilization
    mean = pred_value.mean()
    pred_value = pred_value - mean
    return pred_value 

#Hinge Loss
def hinge_loss(diabetes_outcome, pred_value):
    y_times_pred = pred_value * diabetes_outcome
    #find out all the positive index
    negative_index = y_times_pred.lt(0)
    
    #change all index to (1-y*f(x))
    hinge_loss = 1 - y_times_pred
    
    #replace positive indexs to 0
    hinge_loss = hinge_loss * negative_index
    
    #tada    
    return hinge_loss

#objective function
def objective_function(alpha, w, h_loss):
    
    summ = np.sum(h_loss)
    dott = np.dot(w.T,w)
    minimize = (1/2) * alpha * dott + summ
    return minimize

pred_value = pred_value(w,b,diabetes_parameter)
loss = hinge_loss(diabetes_outcome, pred_value)
minimize = objective_function(alpha, w, loss)


def sgd(diabetes_parameter, diabetes_outcome, diabetes_outcome_withzeros, w, alpha):
    epoch = 10000
    errors = 0
    for epoch in range(1,epoch):
        
        
       #dl_dw = diabetes_outcome.iloc[i] * diabetes_parameter.iloc[i] - (1/epoch) * w
       temp = np.array([diabetes_outcome_withzeros,]*8)
       temp = np.transpose(temp)
       
       dl_dw = temp * diabetes_parameter - (1/epoch) * w       
       
       w = w + alpha * np.sum(dl_dw)
       print('epoch:',epoch, 'w:',w)
    
    return w

best_w = sgd(diabetes_parameter, diabetes_outcome,diabetes_outcome_withzeros, w, alpha)

#%%
test_outcome = test.iloc[:,8]
test_parameter = test.iloc[:,0:8]

np.sum(best_w * test_parameter, axis = 1)














#%%


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
BMI_AGE = diabetes_dataset.iloc[:, [5,7]]
BMI_AGE = np.transpose(BMI_AGE)

#%%
#glucose, BP, skinthickness <- turn 0 to mean?


#%%
#svm model







