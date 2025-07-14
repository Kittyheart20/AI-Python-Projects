# perceptron.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020



import numpy as np

def trainPerceptron(train_set, train_labels,  max_iter):
    #Write code for Mp4
    
    W = np.zeros(train_set.shape[1])
    learning_rate = 0.01
    b = 0
    i=0

    while (i < max_iter):
        for x in range(len(train_set)):
            prediction = np.dot(W, train_set[x]) + b
            #print ("Train Data:", train_set[x], "Prediction: ", prediction, "label:", train_labels[x])
            if ((prediction > 0) and (train_labels[x] != 1)):
                W = W - train_set[x] * learning_rate
                b = b - learning_rate
            elif ((prediction <= 0) and (train_labels[x] != 0)):
                W = W + train_set[x] * learning_rate
                b = b + learning_rate
        i = i + 1

    return W, b

def classifyPerceptron(train_set, train_labels, dev_set, max_iter):
    #Write code for Mp4
    predictions = []

    W,b = trainPerceptron(train_set, train_labels, max_iter)

    for i in range(len(dev_set)):
        if (np.dot(W, dev_set[i]) + b > 0):
            predictions.append(1)
        else:
            predictions.append(0)

    return predictions



