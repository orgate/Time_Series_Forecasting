import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
#import seaborn as sns
import os
import sys

from get_data import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf

#df = pd.DataFrame(np.array([ps_open_t[0],ps_close_t[0],ps_high_t[0],ps_low_t[0]]).T, columns=['open','close','high','low'])


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

TRAIN_TEST_RATIO = 0.66
size = int(len(ps_open_t_arr[0]) * TRAIN_TEST_RATIO)
print("size is: ",size)
#df = df.drop(['symbol','volume'],axis=1) # Drop Adj close and volume feature
df = pd.DataFrame(np.array([ps_open_t_arr[0]]).T, columns=['open'])
dataset = df.values
dataset = dataset.astype('float32')
# normalize the dataset
scaler1 = MinMaxScaler(feature_range=(0, 1))
dataset = scaler1.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * TRAIN_TEST_RATIO)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

#df_train = df[:size]    # 60% training data and 40% testing data
#df_test = df[size:]
scaler = MinMaxScaler() # For normalizing dataset


look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# We want to predict Close value of stock 
X_train = scaler.fit_transform(trainX)
y_train = scaler.fit_transform(trainY.reshape(-1,1))
# y is output and x is features.
X_test = scaler.fit_transform(testX)
y_test = scaler.fit_transform(testY.reshape(-1,1))


def denormalize(df,norm_data):
    df = df.values.reshape(-1,1)
    norm_data = norm_data.reshape(-1,1)
    scl = MinMaxScaler()
    a = scl.fit_transform(df)
    new = scl.inverse_transform(norm_data)
    return new
"""
Above written function for denormalization of data after normalizing
this function will give original scale of values.
In normalization we step down the value of data in dataset.
"""

def neural_net_model(X_data,input_dim):
    W_1 = tf.Variable(tf.random_uniform([input_dim,10]))
    b_1 = tf.Variable(tf.zeros([10]))
    layer_1 = tf.add(tf.matmul(X_data,W_1), b_1)
    layer_1 = tf.nn.relu(layer_1)

    # layer 1 multiplying and adding bias then activation function
    W_2 = tf.Variable(tf.random_uniform([10,10]))
    b_2 = tf.Variable(tf.zeros([10]))
    layer_2 = tf.add(tf.matmul(layer_1,W_2), b_2)
    layer_2 = tf.nn.relu(layer_2)
    # layer 2 multiplying and adding bias then activation function
    W_O = tf.Variable(tf.random_uniform([10,1]))
    b_O = tf.Variable(tf.zeros([1]))
    output = tf.add(tf.matmul(layer_2,W_O), b_O)
    # O/p layer multiplying and adding bias then activation function
    # notice output layer has one node only since performing #regression
    return output
"""
neural_net_model is function applying 2 hidden layer feed forward neural net.
Weights and biases are abberviated as W_1,W_2 and b_1, b_2 
These are variables with will be updated during training.
"""
xs = tf.placeholder("float")
ys = tf.placeholder("float")
output = neural_net_model(xs,look_back)
cost = tf.reduce_mean(tf.square(output-ys))
# our mean squared error cost function
##train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
# Gradinent Descent optimiztion just discussed above for updating weights and biases

correct_pred = tf.argmax(output, 1)
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

c_t = []
c_test = []
Prediction = []

with tf.Session() as sess:
    # Initiate session and initialize all vaiables
    for i in range(y_test.shape[0]-1):
        sess.run(tf.global_variables_initializer())
        train = tf.train.GradientDescentOptimizer(0.001).minimize(cost) ### Added from outside the "with" loop
        saver = tf.train.Saver()
        X_train_list = list(X_train)
        X_train_list.append(X_test[0,:])
        X_train = np.array(X_train_list)   ### Added by Alfred
        y_train_list = list(y_train)
        y_train_list.append(y_test[0])
        y_train = np.array(y_train_list)     ### Added by Alfred
        X_test = X_test[1:,:]                                   ### Added by Alfred
        y_test = y_test[1:]                                     ### Added by Alfred
        #saver.restore(sess,'yahoo_dataset.ckpt')
        for j in range(100):
            #print("X_train.shape[0] is: ",len(X_train))
            for k in range(X_train.shape[0]):
                sess.run([cost,train],feed_dict=    {xs:X_train[k,:].reshape(1,look_back), ys:y_train[k]})
                # Run cost and train with each sample
            c_t.append(sess.run(cost, feed_dict={xs:X_train,ys:y_train}))
            c_test.append(sess.run(cost, feed_dict={xs:X_test[0:1,:],ys:y_test[0:1]}))
            #print('Epoch :',i,'Cost :',c_t[i])
            print('Data: ',i,'Epoch :',j,'Cost :',c_test[j])
        pred = sess.run(output, feed_dict={xs:X_test[0:1,:]})
        Prediction.append(np.concatenate(pred).astype(None))

    Prediction1 = np.array(Prediction)
    #print('Prediction :',Prediction1)
    # predict output of test data after training
    print('Cost :',sess.run(cost, feed_dict={xs:X_test,ys:y_test}))
    #xtest0 = denormalize(df,X_test[:,0])
    #xtest1 = denormalize(df,X_test[:,1])
    #xtest2 = denormalize(df,X_test[:,2])
    y_test1 = scaler.fit_transform(testY.reshape(-1,1))
    y_test1 = y_test1[:-1,:]
    y_test1 = denormalize(df,y_test1)
    Predictions = denormalize(df,Prediction1)
    #Denormalize data     
    #plt.plot(range(y_test.shape[0]),xtest0,label="Original Feature0")    
    #plt.plot(range(y_test.shape[0]),xtest1,label="Original Feature1")    
    #plt.plot(range(y_test.shape[0]),xtest2,label="Original Feature2")    
    plt.plot(range(len(y_test1)),y_test1,label="Original Data")
    plt.plot(range(len(y_test1)),Predictions,label="Predicted Data")
    plt.legend(loc='best')
    plt.ylabel('Stock Value')
    plt.xlabel('Days')
    plt.title('Stock Market Nifty')
    plt.show()
#    if input('Save model ? [Y/N]') == 'Y':
#        saver.save(sess,'yahoo_dataset.ckpt')
#        print('Model Saved')
#error = mean_squared_error(xtest0, pred)
#print('Test MSEx0: %.3f' % error)
#error = mean_squared_error(xtest1, pred)
#print('Test MSEx1: %.3f' % error)
#error = mean_squared_error(xtest2, pred)
#print('Test MSEx2: %.3f' % error)
error = mean_squared_error(y_test1, Predictions)
print('Test MSEy: %.3f' % error)

print("Looks like NN model worked!")





