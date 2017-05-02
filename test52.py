# correlation matrix

import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import cPickle as pickle
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import sys
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv("./speed_code.csv", encoding="ISO-8859-1")
predict_col = 'match'
rem1 = np.array(pickle.load(open(predict_col+'important1removed.p', 'rb')))
input_vars = np.array(pickle.load(open('input_vars_removed.p', 'rb')))
input_vars = np.array(input_vars[rem1])

print input_vars.shape

attribute_num = len(input_vars)
print 'attribute_num', attribute_num, 'predict', predict_col

xs = np.zeros((8378, attribute_num))
ys = np.zeros((8378, 1))

for i in range(attribute_num):
    xs[:, i] = df[input_vars[i]]
ys[:, 0] = df[predict_col]

xs[np.isnan(xs)] = 0.
ys = np.reshape(ys, len(ys))

print xs.shape
print ys.shape

