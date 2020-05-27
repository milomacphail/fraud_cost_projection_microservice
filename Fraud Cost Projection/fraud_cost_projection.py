# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:17:50 2020

@author: milom
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataList = []

# Importing the dataset
for chunk in pd.read_csv("transactionRecords.csv", sep=";", chunksize=200000):
    dataList.append(chunk)
    
dataset = pd.concat(dataList, axis=0)

del dataList

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:, :10].values


# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)