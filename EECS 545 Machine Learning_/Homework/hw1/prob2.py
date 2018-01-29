#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 17:48:15 2018

@author: DavidYang
The script was creatd for problem 2, eecs 545 homwork 1
"""
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

def preprocess(Data):
    ## Step 1 -- Normalization of the Data Set ##
    # First calculate the mean and std of the trainning data set
    Data_mean = np.mean(Data,axis=0,dtype=np.float64)
    Data_std = np.std(Data,axis=0,dtype=np.float64) 
    N = Data.shape[0]
    # Find the column with zero std
    ind_pstd = (Data_std != 0)
    # Normalize the trainning data set by substracting the mean value
    Data = Data - Data_mean
    # Normalize the feature with std if std is not zero
    Data[:,ind_pstd] = Data[:,ind_pstd]/Data_std[ind_pstd]
    # Add bias term to the data set
    bias_col_train = np.ones((N,1))
    Data_norm = np.append(bias_col_train,Data,axis=1)
    return Data_norm

# Define the function to compute root mean squared error:
def rmse(X,y,w):
    return np.sqrt(np.sum((np.dot(np.transpose(w),np.transpose(X)) - np.transpose(y))**2)/len(y))

# Create a function generating polynomial features to existing features
def GeneratePoly(features_orig,order):
    n_cols = features_orig.shape[1]
    n_rows = features_orig.shape[0]
    features = []
    if order == 0:
        features = np.ones((n_rows,1))
    elif order == 1:
        features = features_orig
    else: 
        features = features_orig
        for col in range(n_cols):
            for cur_ord in range(2,order+1):
                z = np.power(features_orig[:,col],cur_ord).reshape(n_rows,1)
                features = np.append(z, features, axis=1)
    return features

# Create a solver function using closed form method:
def solver_cl(X,y):
    return np.dot(np.linalg.pinv(X),y)

# Create a data set split function
def DataSplit(features,labels,p):
    NData = features.shape[0]
    Nsplit = int(NData*(1-p)-1)
    X_train, y_train = features[:-Nsplit], labels[:-Nsplit]
    X_test, y_test = features[-Nsplit:], labels[-Nsplit:]
    return {'X_train':X_train,'y_train':y_train,'X_test':X_test,'y_test':y_test}

############################# Part(a) ########################################
# Load dataset
dataset = datasets.load_boston()
features = dataset.data
labels = dataset.target

# Define features for different polynomials
features_0 = GeneratePoly(features,0)
features_1 = GeneratePoly(features,1)
features_2 = GeneratePoly(features,2)
features_3 = GeneratePoly(features,3)
features_4 = GeneratePoly(features,4)
# Preprocess all the features
features_1 = preprocess(features_1)
features_2 = preprocess(features_2)
features_3 = preprocess(features_3)
features_4 = preprocess(features_4)

# Split the data into x_train and x_test
# Split dataset
Nsplit = 50
# Training set
X_train_0, y_train = features_0[:-Nsplit], labels[:-Nsplit]
X_train_1 = features_1[:-Nsplit]
X_train_2 = features_2[:-Nsplit]
X_train_3 = features_3[:-Nsplit]
X_train_4 = features_4[:-Nsplit]
# Test set
X_test_0, y_test = features_0[-Nsplit:], labels[-Nsplit:]
X_test_1 = features_1[-Nsplit:]
X_test_2 = features_2[-Nsplit:]
X_test_3 = features_3[-Nsplit:]
X_test_4 = features_4[-Nsplit:]

# Find closed form solution
w_cl0 = solver_cl(X_train_0,y_train)
w_cl1 = solver_cl(X_train_1,y_train)
w_cl2 = solver_cl(X_train_2,y_train)
w_cl3 = solver_cl(X_train_3,y_train)
w_cl4 = solver_cl(X_train_4,y_train)

# Compute the test error for each case
err_0 = rmse(X_test_0,y_test,w_cl0)
err_1 = rmse(X_test_1,y_test,w_cl1)
err_2 = rmse(X_test_2,y_test,w_cl2)
err_3 = rmse(X_test_3,y_test,w_cl3)
err_4 = rmse(X_test_4,y_test,w_cl4)
error_test = np.array([err_0,err_1,err_2,err_3,err_4])

# Compute the train error for each case
err_0 = rmse(X_train_0,y_train,w_cl0)
err_1 = rmse(X_train_1,y_train,w_cl1)
err_2 = rmse(X_train_2,y_train,w_cl2)
err_3 = rmse(X_train_3,y_train,w_cl3)
err_4 = rmse(X_train_4,y_train,w_cl4)
error_train = np.array([err_0,err_1,err_2,err_3,err_4])

# plot
plt.figure(1)
train_error_line ,= plt.plot(range(5),error_train,label='train error')
test_error_line ,= plt.plot(range(5),error_test,label='test error')
plt.xlabel('Percentage of training set in terms of total data set')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.legend([train_error_line, test_error_line])
plt.xlabel('Order of Polynomials')
plt.ylabel('Root Mean Squared Error (RMSE)')

############################## Part (b) ######################################
NofSplit = 5
per = np.linspace(0.2,1,NofSplit)
train_err = []
test_err = []
for i in range(NofSplit):
    result = DataSplit(features_1,labels,per[i])
    X_train = result['X_train']
    y_train = result['y_train']
    X_test = result['X_test']
    y_test = result['y_test']
    w_cl = solver_cl(X_train,y_train)
    print(np.array_str(np.array(w_cl), precision=2))
    train_err.append(rmse(X_train,y_train,w_cl))
    test_err.append(rmse(X_test,y_test,w_cl))

plt.figure(2)
train_error_line ,= plt.plot(per*100,train_err,label='train error')
test_error_line ,= plt.plot(per*100,test_err,label='test error')
plt.xlabel('Percentage of training set in terms of total data set')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.legend([train_error_line, test_error_line])
