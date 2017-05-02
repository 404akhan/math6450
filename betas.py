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

df = pd.read_csv("./speed_code.csv", encoding="ISO-8859-1")
predict_col = 'match'
rem1 = np.array(pickle.load(open(predict_col+'important1removed.p', 'rb')))
input_vars = np.array(pickle.load(open('input_vars_removed.p', 'rb')))
input_vars = np.array(input_vars[rem1])
attribute_num = len(input_vars)
print 'attribute_num', attribute_num, 'predict', predict_col

xs = np.zeros((8378, attribute_num))
ys = np.zeros((8378, 1))

for i in range(attribute_num):
    xs[:, i] = df[input_vars[i]]
ys[:, 0] = df[predict_col]

xs[np.isnan(xs)] = 0.
ys = np.reshape(ys, len(ys))

X_train, X_test, y_train, y_test = train_test_split(
    xs, ys, test_size=0.33, random_state=42)

xs_mean = np.mean(X_train, axis=0)
xs_std = np.std(X_train, axis=0)

X_train = (X_train - xs_mean) / xs_std
X_test = (X_test - xs_mean) / xs_std

second_row = []

rem4 = None

def apply_model(clf):
    global second_row, rem4
    clf.fit(X_train, y_train)

    predict = clf.predict(X_test)
    print predict
    error = np.mean(np.square(predict - y_test))
    print '%.2f%%' % ((1 - error) * 100)
    second_row.append('%.2f%%' % ((1 - error) * 100))
    print clf.coef_
    rem4, = clf.coef_
    return predict

models = {}
models['LogisticRegression'] = LogisticRegression()

first_row = []

rem = np.zeros_like(y_test)
num_models = 0
for k, v in models.iteritems():
    print '\nmodel: ' + k
    first_row.append(k)
    rem += apply_model(v)
    num_models += 1

rem /= num_models
rem = (rem > 0.5).astype(np.int)

error = np.mean(np.square(rem - y_test))
# print '\nmodel: Ensemble'
# print '%.2f%%' % ((1 - error) * 100)

first_row.append('Ensemble')
second_row.append('%.2f%%' % ((1 - error) * 100))

# print '\n\n'
# print input_vars
# print rem4

dict = {}

for i in range(len(input_vars)):
    dict[input_vars[i]] = rem4[i]

import operator

sorted_x = sorted(dict.items(), key=operator.itemgetter(1))

print '\n\n' + predict_col
for k, v in sorted_x[::-1]:
	print k, v
with open('results-removed-90/' + predict_col+'-betas-normalized.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for k in sorted_x[::-1]:
        spamwriter.writerow([k[0], k[1]])

