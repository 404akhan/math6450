import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

df = pd.read_csv("./speed_code.csv", encoding="ISO-8859-1")
input_vars = np.array(['attr', 'sinc', 'intel', 'fun', 'amb', 'shar', 'like', 'prob',
                       'attr_o', 'sinc_o', 'intel_o', 'fun_o', 'amb_o', 'shar_o', 'like_o', 'prob_o'])

attribute_num = len(input_vars)
print 'attribute_num', attribute_num

xs = np.zeros((8378, attribute_num))
ys = np.zeros((8378, 1))

for i in range(attribute_num):
    xs[:, i] = df[input_vars[i]]
ys[:, 0] = df['dec_o']

xs[np.isnan(xs)] = 6.
ys = np.reshape(ys, len(ys))

X_train, X_test, y_train, y_test = train_test_split(
    xs, ys, test_size=0.33, random_state=42)

def apply_model(clf):
    clf.fit(X_train, y_train)

    predict = clf.predict(X_test)
    print predict
    error = np.mean(np.square(predict - y_test))
    print error
    return predict

def apply_model_div5(clf):
    clf.fit(xs_train, ys_train)

    predict = clf.predict(xs_cross_val)
    error = np.mean(np.square(predict - ys_cross_val))

    return error

models = {}
models['LogisticRegression'] = LogisticRegression()
models['SVC'] = SVC()
models['LinearDiscriminantAnalysis'] = LinearDiscriminantAnalysis()

random.seed(133)
shuf_arr = range(0, 8378)
random.shuffle(shuf_arr)
div5 = 8378/5

sum_error = {k: 0. for k in models}
for k, v in models.iteritems():
    for i in range(5):
        if i != 4:
            xs_train = xs[shuf_arr[div5*i:div5*(i+1)], :]
            xs_cross_val = xs[shuf_arr[div5*(i+1):], :]
            ys_train = ys[shuf_arr[div5*i:div5*(i+1)]]
            ys_cross_val = ys[shuf_arr[div5*(i+1):]]

            sum_error[k] += apply_model_div5(v)
        else:
            xs_train = xs[shuf_arr[div5*(i):], :]
            xs_cross_val = xs[shuf_arr[:div5*(i)], :]
            ys_train = ys[shuf_arr[div5*(i):]]
            ys_cross_val = ys[shuf_arr[:div5*(i)]]

            sum_error[k] += apply_model_div5(v)

for k, v in sum_error.iteritems():
    print k, v / 5
