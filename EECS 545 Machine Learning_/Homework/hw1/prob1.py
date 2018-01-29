import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Define the function for data preprocess:
def preprocess(Data):
    ## Step 1 -- Normalization of the Data Set ##
    # First calculate the mean and std of the trainning data set
    Data_mean = np.mean(Data,axis=0,dtype=np.float64)
    Data_std = np.std(Data,axis=0,dtype=np.float64) 
    N = len(Data[:,1])
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

# Define the function to compute mean squared error:
def mse(X,y,w):
    return np.sum((np.dot(np.transpose(w),np.transpose(X)) - np.transpose(y))**2)/len(y)

# Load dataset
dataset = datasets.load_boston()
features = dataset.data
labels = dataset.target

# 1. Preprocess the entire data set
features = preprocess(features)

# Split dataset
Nsplit = 50
# Training set
X_train, y_train = features[:-Nsplit], labels[:-Nsplit]
# Test set
X_test, y_test = features[-Nsplit:], labels[-Nsplit:]

# 2. Initialize the weight vector w
w_sgd = np.random.uniform(low=-0.1,high=0.1,size=14)

# 3. Initialize the learning rate eta
eta = 0.05
dev = 1
epoch = 0
N_epoch = 500
train_error = np.array(np.ones(N_epoch))
test_error = np.array(np.ones(N_epoch))
# 4a. Apply the SGD wth a while loop 
while (epoch < N_epoch):
    # 5a.Shuffle the training data set
    seed = np.arange(len(y_train))
    np.random.shuffle(seed)
    X_train = X_train[seed]
    y_train = y_train[seed]
    # 6a. Apply the update rule to the weight vector
    for i in range(len(y_train)):
        dev = 2.0/len(y_train)*(np.dot(np.transpose(w_sgd),np.transpose(X_train[i,:])) - y_train[i])* np.transpose(X_train[i,:])
        w_sgd = w_sgd - eta*dev
    # 7a. Compute the training error for the current epoch    
    train_error[epoch] = mse(X_train,y_train,w_sgd)
    test_error[epoch] = mse(X_test,y_test,w_sgd)
    epoch += 1
print('Training error [SGD]',train_error[-1], 'Testing Error [SGD]',test_error[-1])    

plt.figure(1)   
train_error_line, = plt.plot(range(N_epoch),train_error[range(N_epoch)],label='train error')
test_error_line, = plt.plot(range(N_epoch),test_error[range(N_epoch)],label='test error')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE) -- SGD')
plt.legend([train_error_line, test_error_line])

################ Part 2 Batch Gradient Decent Algorithm ######################
# 1. Shuffle the data set
seed = np.arange(len(y_train))
np.random.shuffle(seed)
X_train = X_train[seed]
y_train = y_train[seed]

# 2. Initialize the weight vector w
w_bgd = np.random.uniform(low=-0.1,high=0.1,size=14)

# 3. Initialize the learning rate eta
eta = 0.05
dev = 1
epoch = 0
N_epoch = 500
train_error = np.array(np.ones(N_epoch))
test_error = np.array(np.ones(N_epoch))
# 4b. Apply the Batch Gradient Decent Algorithm 
while (epoch < N_epoch):
    dev = 2.0/len(y_train)*np.dot((np.dot(np.transpose(w_bgd),np.transpose(X_train)) - np.transpose(y_train)),X_train)
    w_bgd = w_bgd - eta*dev
    #print(dev)
    # 7a. Compute the training error for the current epoch    
    train_error[epoch] = mse(X_train,y_train,w_bgd)
    test_error[epoch] = mse(X_test,y_test,w_bgd)
    epoch += 1

print('Training error [BGD]',train_error[-1], 'Testing Error [BGD]',test_error[-1])    
plt.figure(2) 
train_error_line, = plt.plot(range(N_epoch),train_error[range(N_epoch)],label='train error')
test_error_line, = plt.plot(range(N_epoch),test_error[range(N_epoch)],label='test error')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE) -- BGD')
plt.legend([train_error_line, test_error_line])

############### Part 3 Closed Form Solution ######################################
w_cl = np.dot(np.linalg.pinv(X_train),y_train)
train_error_cl = mse(X_train,y_train,w_cl)
test_error_cl = mse(X_test,y_test,w_cl)

print('Training error [Closed Form]',train_error_cl, 'Testing Error [Closed Form]',test_error_cl)

############### Part 4 Random Split of the data set ##############################

train_errs = []
test_errs = []
features_orig = dataset.data
labels_orig = dataset.target

for k in range(100):
    # Original features
    # seed = np.arange(len(labels_orig))
    
    # Shuffle data
    #np.random.shuffle(seed)
    features = features_orig
    #labels = labels_orig[seed]
    
    features = preprocess(features)
    Nsplit = np.random.randint(low=10,high=400)
    # Training set
    X_train, y_train = features[:-Nsplit], labels[:-Nsplit]
    # Test set
    X_test, y_test = features[-Nsplit:], labels[-Nsplit:]
    
    # Solve for optimal w
    w_cle = np.dot(np.linalg.pinv(X_train),y_train)
    
    # Collect train and test errors
    train_error_cl = mse(X_train,y_train,w_cle)
    test_error_cl = mse(X_test,y_test,w_cle)
    
    train_errs.append(train_error_cl)
    test_errs.append(test_error_cl)

print('Mean training error: ', np.mean(train_errs))
print('Mean test error: ', np.mean(test_errs))
