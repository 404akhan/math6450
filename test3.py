import numpy as np
import pandas as pd
import random
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
import sys
from pylab import pcolor, show, colorbar, xticks, yticks
import cPickle as pickle
from sklearn.model_selection import train_test_split

df = pd.read_csv("./speed_code.csv", encoding="ISO-8859-1")
input_vars = np.array(['attr', 'sinc', 'intel', 'fun', 'amb', 'shar', 'like', 'prob',
                       'attr_o', 'sinc_o', 'intel_o', 'fun_o', 'amb_o', 'shar_o', 'like_o', 'prob_o',
                       'samerace'])

input_vars = np.array([u'condtn', u'round', u'samerace', u'attr_o', u'sinc_o', u'fun_o', u'amb_o',
                       u'like_o', u'prob_o', u'met_o', u'date', u'go_out', u'museums', u'art',
                       u'amb3_1', u'attr', u'fun', u'amb', u'like', u'prob'])

attribute_num = len(input_vars)
print 'attribute_num', attribute_num

xs = np.zeros((8378, attribute_num))
ys = np.zeros((8378, 1))

for i in range(attribute_num):
    xs[:, i] = df[input_vars[i]]
ys[:, 0] = df['match']

xs[np.isnan(xs)] = 0.

xs_train, xs_cross_val, ys_train, ys_cross_val = train_test_split(
    xs, ys, test_size=0.33, random_state=42)

train_size = xs_train.shape[0]
lr = 0.01

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def get_loss():
    scores = np.matmul(xs_cross_val, W) + b
    predict = sigmoid(scores)

    error = predict - ys_cross_val
    return np.mean(np.square(error))

def get_accuracy():
    scores = np.matmul(xs_cross_val, W) + b
    predict = sigmoid(scores)
    predict = (predict > 0.5).astype(np.int)

    error = predict - ys_cross_val
    return np.mean(np.abs(error))

def get_train_loss():
    scores = np.matmul(xs_train, W) + b
    predict = sigmoid(scores)

    error = predict - ys_train
    return np.mean(np.square(error))

def get_train_accuracy():
    scores = np.matmul(xs_train, W) + b
    predict = sigmoid(scores)
    predict = (predict > 0.5).astype(np.int)

    error = predict - ys_train
    return np.mean(np.abs(error))

W = 0.01 * np.random.randn(attribute_num, 1)
b = 0.

for i in range(100*1000):
    scores = np.matmul(xs_train, W) + b
    predict = sigmoid(scores)

    dpredict = 1. / train_size * (predict - ys_train)
    dscores = dpredict * predict * (1 - predict)
    dW = np.matmul(xs_train.transpose(), dscores)
    db = np.sum(dscores)

    W -= lr * dW
    b -= lr * db

    if i % 10 == 0:
        print 'iter: %d, loss: %f, acc: %f, tr loss: %f, tr acc: %f' % (i, get_loss(), get_accuracy(), get_train_loss(), get_train_accuracy())
