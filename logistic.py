import numpy as np
import pandas as pd
import random
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
import sys
from pylab import pcolor, show, colorbar, xticks, yticks

df = pd.read_csv("./speed_code.csv", encoding="ISO-8859-1")
input_vars = np.array(['attr', 'sinc', 'intel', 'fun', 'amb', 'shar', 'like', 'prob',
                       'attr_o', 'sinc_o', 'intel_o', 'fun_o', 'amb_o', 'shar_o', 'like_o', 'prob_o',
                       'samerace'])

attribute_num = 3*len(input_vars)
print 'attribute_num', attribute_num

xs = np.zeros((8378, attribute_num))
ys = np.zeros((8378, 1))

for i in range(attribute_num/3):
    xs[:, i] = df[input_vars[i]]
    xs[:, i + attribute_num/3] = xs[:, i]*xs[:, i]
    xs[:, i + 2*attribute_num/3] = xs[:, i]*xs[:, i]*xs[:, i]
ys[:, 0] = df['dec_o']

xs[np.isnan(xs)] = 6.

random.seed(1339)
shuf_arr = range(0, 8378)
random.shuffle(shuf_arr)
train_size = int(8378 * 0.7)
lr = 0.03

xs_train = xs[shuf_arr[0:train_size], :]
xs_cross_val = xs[shuf_arr[train_size:], :]
ys_train = ys[shuf_arr[0:train_size], :]
ys_cross_val = ys[shuf_arr[train_size:], :]

xs_mean = np.mean(xs_train, axis=0)
xs_std = np.std(xs_train, axis=0)

xs_train = (xs_train - xs_mean) / xs_std
xs_cross_val = (xs_cross_val - xs_mean) / xs_std

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

