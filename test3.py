import numpy as np
import pandas as pd
import random
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
import sys

df = pd.read_csv("./speed_code.csv", encoding="ISO-8859-1")
input_vars = np.array(['attr', 'sinc', 'intel', 'fun', 'amb', 'shar', 'like', 'prob',
                       'attr_o', 'sinc_o', 'intel_o', 'fun_o', 'amb_o', 'shar_o', 'like_o', 'prob_o',
                       'samerace'])

attribute_num = len(input_vars)
print 'attribute_num', attribute_num

xs = np.zeros((8378, attribute_num))
ys = np.zeros((8378, 1))

for i in range(attribute_num):
    xs[:, i] = df[input_vars[i]]
ys[:, 0] = df['dec_o']

xs[np.isnan(xs)] = 0.
ys[np.isnan(ys)] = 0.

random.seed(1339)
shuf_arr = range(0, 8378)
random.shuffle(shuf_arr)
train_size = int(8378 * 0.7)
lr = 0.01

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

def get_loss(W, b):
    scores = np.matmul(xs_cross_val, W) + b
    predict = sigmoid(scores)

    error = predict - ys_cross_val
    return np.mean(np.square(error))

def get_accuracy(W, b):
    scores = np.matmul(xs_cross_val, W) + b
    predict = sigmoid(scores)
    predict = (predict > 0.5).astype(np.int)

    error = predict - ys_cross_val
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
        print 'iter: %d, loss: %f, accuracy: %f' % (i, get_loss(W, b), get_accuracy(W, b))
