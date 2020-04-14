#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 20:14:00 2019

@author: jamesyang
"""

import numpy as np

while True:
    try:
        #case = 1
        case = int(input('[STEP 1] Enter the index of data(1 or 2): '))
        #case = int(input('please enter the Case number(Enter 1 or 2): '))
        break
    except ValueError:
        print('Please enter number, this is not a number!')
        
    

if case == 1:           #This is data1 case
    data = np.loadtxt(open(
        './data1(1).txt', 'r'), delimiter="\t", dtype=float)
    X = data[:,0]               # data of x
    Y = data[:,-1]              #data of y
    W = np.array([ [19],[3] ])  # W initialize

elif case == 2:         #This is data2 case
    data = np.loadtxt(open(
        './data2(1).txt', 'r'), delimiter="\t", dtype=float)
    X = data[:, :-1]    # input data
    Y = data[:, -1]         # output data
    W = np.array([ [0],[0],[0],[0],[0] ])     # W initialize
    
    
    
# =============================================================================
# f = open('/Users/jamesyang/Downloads/ML/data1.txt', 'r')
# 
# data = np.loadtxt(f, delimiter="\t", dtype=float)
# 
# X = data[:,0] # sample of x
# Y = data[:,-1] #the answer sample of y
# 
# W = np.array([ [19],[3] ])   #W参數初始化
# =============================================================================

lr = 0.000001   #learning rate 
max_epoch = 10000   #最大迭代次數 
error = 0            #error  
epoch_count = 0      #當前迭代次數  


def Prediction(x):
    for i in range(len(W)):
        if i == 0:
            prediction_y = W[i][0]*1 # x0=1
        else:
            prediction_y += W[i][0]*x[i-1]

    return prediction_y


while(epoch_count < max_epoch):     #epoch_count < max_epochStop the epoch until do the "max_epoch" times
    error = 0 
    m = len(X)
    for i in range(m):
        if case == 1:
            x = np.array([[1],[X[i]]]).reshape(1,2)
        elif case == 2:
            x_insert = np.insert(X[i],0,1)
            x = np.array([x_insert]).reshape(1,5)
            
        y = np.array([Y[i]]).reshape(1,1)
        
        pred_y = np.dot(x,W)                    #WTx
        W = W + lr * (x.T.dot((y - pred_y)))    # W <- W + learning_rate * (y - y^)x
        #print(W)
        error += (y - pred_y)**2
    mse = (1/m) * error
    print("MSE: ",mse)
    print('epoch_count: ',epoch_count)
    print("==================")
    #if mse < 1000:
     #   break
    epoch_count += 1 



print('<--------(%s)----------->'%(case))
print ('Total epoch: ', epoch_count) 
print ('MSE: ', mse)  

print('-------(%s)(a)---------'%(case))
print ('W : ',W )
for i in range(len(W)):
    print('W%s: %d'%(i, W[i]))
    
print('-------(%s)(b)---------'%(case))
if case == 1:
    print('Predict x=45. y= ',Prediction([45]))
    print('Predict x=25. y= ',Prediction([25]))
elif case == 2:
    print('Predict (x1, x2, x3, x4) = (6.8, 210, 0.402, 0.739), y= ',
        Prediction([6.8, 210, 0.402, 0.739]))
    print('Predict (x1, x2, x3, x4) = (6.1, 180, 0.415, 0.713), y= ',
        Prediction([6.1, 180, 0.415, 0.713]))









