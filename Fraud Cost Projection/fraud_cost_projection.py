# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:17:50 2020

@author: milom
"""

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd

#import Flask
from Flask import Flask, request
from Flask_json import FlaskJSON, JsonError, json_response

#CREATING DATA MODEL

#Create list to aid in batch import
dataList = []

# Importing the dataset
for chunk in pd.read_csv("transactionRecords.csv", sep=";", chunksize=200000):
    dataList.append(chunk)
    
dataset = pd.concat(dataList, axis=0)

del dataList

#creating scaler
from sklearn.preprocessing import StandardScaler
columns = ['amount', 'oldBalanceOrig', 'newBalanceOrig' , 'oldBalanceDest', 'newBalanceDest']
schema = dataset[columns]
scaler = StandardScaler().fit(schema.values)
schema = scaler.transform(schema.values)
dataset[columns] = schema

X = dataset.iloc[:, [1, 2, 4, 5, 7, 8]].values
y = dataset.iloc[:, 9].values

#Handle categorical variables
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
A = make_column_transformer(
    (OneHotEncoder(categories='auto'), [0]),
    remainder = 'passthrough')

X=A.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""


