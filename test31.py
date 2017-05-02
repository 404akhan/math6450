#coef_
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import cPickle as pickle
import operator

df = pd.read_csv("./speed_code.csv", encoding="ISO-8859-1")
# input_vars = np.array(['attr', 'sinc', 'intel', 'fun', 'amb', 'shar', 'like', 'prob',
#                        'attr_o', 'sinc_o', 'intel_o', 'fun_o', 'amb_o', 'shar_o', 'like_o', 'prob_o'])
#
# mask = pickle.load(open('important1.p', 'rb'))
# input_vars = pickle.load(open('input_vars.p', 'rb'))
# input_vars = np.array(input_vars)[mask]

input_vars = np.array(['condtn', 'round', 'samerace', 'attr_o', 'sinc_o', 'fun_o', 'amb_o',
                       'like_o', 'prob_o', 'met_o', 'date', 'go_out', 'museums', 'art',
                       'amb3_1', 'attr', 'fun', 'amb', 'like', 'prob', 'match_es', 'attr3_s',
                       'sinc3_s', 'fun3_s', 'amb3_s', 'attr5_2', 'them_cal', 'date_3',
                       'num_in_3', 'attr3_3', 'intel3_3', 'fun3_3', 'amb3_3'])

rem1 = pickle.load(open('important1match_miss1000.p', 'rb'))

input_vars = pickle.load(open('input_vars_miss1000.p', 'rb'))

input_vars = input_vars[rem1]

# print input_vars[0]
# input_vars = input_vars[1:]

attribute_num = len(input_vars)
print 'attribute_num', attribute_num

xs = np.zeros((8378, attribute_num))
ys = np.zeros((8378, 1))

for i in range(attribute_num):
    xs[:, i] = df[input_vars[i]]
ys[:, 0] = df['match']

xs[np.isnan(xs)] = 0.
ys = np.reshape(ys, len(ys))

X_train, X_test, y_train, y_test = train_test_split(
    xs, ys, test_size=0.33, random_state=42)

coef = None

def apply_model(clf):
    global coef
    clf.fit(X_train, y_train)

    predict = clf.predict(X_test)
    print predict
    error = np.mean(np.square(predict - y_test))
    print error
    coef,  = clf.coef_
    return predict

models = {}
models['LogisticRegression'] = LogisticRegression()

rem = np.zeros_like(y_test)
num_models = 0
for k, v in models.iteritems():
    print '\nmodel: ' + k
    rem += apply_model(v)
    num_models += 1

dict = {}
for i in range(len(input_vars)):
    dict[input_vars[i]] = np.abs(coef[i])
    print input_vars[i], coef[i]

sorted_x = sorted(dict.items(), key=operator.itemgetter(1))

print '\n\nSorted'
for k in sorted_x[::-1]:
    print k[0], dict[k[0]]
