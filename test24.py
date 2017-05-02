# divide data into gender == 0 and gender == 1

import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

df = pd.read_csv("./speed_code.csv", encoding="ISO-8859-1")
input_vars = np.array(['attr', 'sinc', 'intel', 'fun', 'amb', 'shar', 'like', 'prob',
                       'attr_o', 'sinc_o', 'intel_o', 'fun_o', 'amb_o', 'shar_o', 'like_o', 'prob_o', 'gender'])

attribute_num = len(input_vars)
print 'attribute_num', attribute_num

XS = np.zeros((8378, attribute_num))
for i in range(attribute_num):
    XS[:, i] = df[input_vars[i]]

count_0, count_1 = 0, 0
arr_0, arr_1 = [], []
for i in range(XS.shape[0]):
    if XS[i, 16] == 0:
        count_0 += 1
        arr_0.append(i)
    if XS[i, 16] == 1:
        count_1 += 1
        arr_1.append(i)
print count_0, count_1
print count_0 + count_1

xs_g = [XS[arr_0, :16], XS[arr_1, :16]]

YS = np.zeros(8378)
YS[:] = df['dec_o']

ys_g = [YS[arr_0], YS[arr_1]]

models = {}
models['LogisticRegression'] = LogisticRegression()
models['SVC'] = SVC()
models['LinearDiscriminantAnalysis'] = LinearDiscriminantAnalysis()

rem = {k: 0. for k in models}
size_of_test = [0, 0]
for g in range(2):
    print 'gender: %d' % g
    xs = xs_g[g]
    ys = ys_g[g]

    xs[np.isnan(xs)] = 6.

    X_train, X_test, y_train, y_test = train_test_split(
        xs, ys, test_size=0.33, random_state=42)
    size_of_test[g] = len(y_test)

    def apply_model(clf):
        clf.fit(X_train, y_train)

        predict = clf.predict(X_test)
        error = np.sum(np.abs(predict - y_test))
        return error

    for k, v in models.iteritems():
        rem[k] += apply_model(v)

for k in rem:
    print k, rem[k]/np.sum(size_of_test)