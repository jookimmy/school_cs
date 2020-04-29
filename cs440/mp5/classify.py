# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set
"""

import numpy as np

def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters 
    
    w = np.zeros(train_set[0].size)
    b = 0
    
    for epoch in range(max_iter):
        for idx, item in enumerate(train_set):
            ystar = np.sign((w @ item) + b)
            y = 1 if train_labels[idx] else -1
            if y != ystar:
                w = w + (learning_rate * y * item)
                b = b + (learning_rate * y)

    return w, b

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train perceptron model and return predicted labels of development set
    w, b = trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    dev_set = np.transpose(dev_set)
    yhats = [1 if yhat == 1 else 0 for yhat in np.sign((w @ dev_set) + b)]
    return yhats

def sigmoid(x):
    # TODO: Write your code here
    # return output of sigmoid function given input x
    return 1/(1 + np.exp(-x))

def trainLR(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters 
    
    weights = np.zeros(train_set[0].size + 1)
    N = len(train_set)
    

    for epoch in range(max_iter):
        loss = 0
        for idx, item in enumerate(train_set):
            features = np.append(item, 1)
            yhat = sigmoid(np.dot(weights,features))
            y = train_labels[idx]
            
            dLds = (y/yhat) - ((1-y)/(1-yhat))
            dsdw = (yhat - yhat**2) * features
            loss += dLds * dsdw
        
        weights -= (-1/N) * learning_rate * loss
    
    return weights[:-1], weights[-1]

def classifyLR(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train LR model and return predicted labels of development set
    weights, bias = trainLR(train_set, train_labels, learning_rate, max_iter)
    return [round(yhat) for yhat in sigmoid(np.dot(weights,np.transpose(dev_set)) + bias)]


def classifyEC(train_set, train_labels, dev_set, k):
    # Write your code here if you would like to attempt the extra credit
    return []
