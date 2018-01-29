# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 22:13:32 2018

@author: david
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
    features = features_orig
    if order == 0:
        features = np.ones((n_rows,1))
    elif order == 1:
        features = np.append(np.ones((n_rows,1)),features,axis=1)
    else: 
        for col in range(n_cols):
            for cur_ord in range(2,order+1):
                z = np.power(features_orig[:,col],cur_ord).reshape(n_rows,1)
                features = np.append(z, features, axis=1)
        features = np.append(np.ones((n_rows,1)),features,axis=1)
    return features

# Create a solver function using closed form method:
def solver_cl(X,y,la):
    n_rows = X.shape[0]
    n_cols = X.shape[1]
    left = np.linalg.inv(np.dot(X.T,X) + n_rows*la*np.eye(n_cols)) 
    right = np.dot(X.T,y)
    return np.dot(left,right)

# Create a data set split function
def DataSplit_1(features,labels):
    Nsplit = 50
    X_train, y_train = features[:-Nsplit], labels[:-Nsplit]
    X_test, y_test = features[-Nsplit:], labels[-Nsplit:]
    return {'X_train':X_train,'y_train':y_train,'X_test':X_test,'y_test':y_test}

def DataSplit_2(features,labels,p):
    NData = features.shape[0]
    Nsplit = int(NData*(1-p)-1)
    X_train, y_train = features[:-Nsplit], labels[:-Nsplit]
    X_val, y_val = features[-Nsplit:], labels[-Nsplit:]
    return {'X_train':X_train,'y_train':y_train,'X_val':X_val,'y_val':y_val}

################################# Part (b) ########################################
# The array for different hyperparameter
lamb = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])

# Load dataset
dataset = datasets.load_boston()
features = dataset.data
labels = dataset.target

# Preprocess
features = preprocess(features)

# Split the train data set and test data set
result_1 = DataSplit_1(features,labels)
X_train = result_1['X_train']
y_train = result_1['y_train']
X_test = result_1['X_test']
y_test = result_1['y_test']

# Split the train data set with the validation set
result_2 = DataSplit_2(X_train, y_train, 0.9)
X_train = result_2['X_train']
y_train = result_2['y_train']
X_val = result_2['X_val']
y_val = result_2['y_val']
# Compute the solution for each hyperparameter case and store the error
train_err = []
val_err = []
w_cl = []
for la in lamb:
    w_cl = solver_cl(X_train,y_train,la)
    train_err.append(rmse(X_train,y_train,w_cl))
    val_err.append(rmse(X_val,y_val,w_cl))
    
plt.figure(1)
train_error_line ,= plt.plot(lamb,train_err,label='train error')
val_error_line ,= plt.plot(lamb,val_err,label='validation error')
plt.xlabel('Hyperparameter $\lambda$')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.legend([train_error_line, val_error_line])

# After selection, we notice that 0.1 is the best hyper-parameter to pick
la_best = 0.1
w_cl = solver_cl(X_train,y_train,la_best)
test_err = rmse(X_test,y_test,w_cl)
print("The test error is:", test_err)