#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 19:24:33 2023

@author: jing
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set()

iris = sns.load_dataset('iris')

iris.head()

#setosa = iris.loc[iris.species == 'setosa']
#virginica = iris.loc[iris.species == 'virginica']

#setosa.sepal_length.plot.hist()

#sns.histplot(setosa.sepal_length, color = 'r')
#sns.histplot(virginica.sepal_length, color = 'b')


import os
abspath = os.path.abspath(__file__) #current file path
dname = os.path.dirname(abspath)#the directory of this file, dataset is also here
os.chdir(dname)

#%%
diabetes_dataset = pd.read_csv('diabetes.csv')
df = diabetes_dataset

#mean normalization
normalized_df=(df-df.mean())/df.std()

#minmac normalization
normalized_df2=(df-df.min())/(df.max()-df.min())

#sns.displot(normalized_df.iloc[:,0])
#plt.savefig('test.png')

#sns.kdeplot(normalized_df)

#sns.pairplot(normalized_df)

#sns.boxenplot(normalized_df)

sns.heatmap(diabetes_dataset)


#%%
output = str('output')
os.chdir(output)
#for i in range(9):
#        sns.displot(normalized_df.iloc[:,i])
#        cate = diabetes_dataset.columns[i]
#        plt.savefig(cate+'_normdist')




