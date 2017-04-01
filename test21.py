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

models = {}
models['LogisticRegression'] = LogisticRegression()
models['SVC'] = SVC()
models['LinearDiscriminantAnalysis'] = LinearDiscriminantAnalysis()
models['QuadraticDiscriminantAnalysis'] = QuadraticDiscriminantAnalysis()

rem = np.zeros_like(y_test)
num_models = 0
for k, v in models.iteritems():
    print '\nmodel: ' + k
    rem += apply_model(v)
    num_models += 1

rem /= 3
rem = (rem > 0.5).astype(np.int)

error = np.mean(np.square(rem - y_test))
print '\nmodel: Ensemble'
print error

