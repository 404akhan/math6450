import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import cPickle as pickle

df = pd.read_csv("./speed_code.csv", encoding="ISO-8859-1")
input_vars = np.array(['attr', 'sinc', 'intel', 'fun', 'amb', 'shar', 'like', 'prob',
                       'attr_o', 'sinc_o', 'intel_o', 'fun_o', 'amb_o', 'shar_o', 'like_o', 'prob_o'])

# input_vars = np.array(['like_o', 'attr_o', 'prob_o', 'them_cal', 'prob', 'amb_o' ,'sinc_o' ,'attr'])
input_vars = np.array(['condtn', 'int_corr', 'samerace', 'attr_o', 'like_o', 'prob_o',
                       'met_o', 'attr', 'like', 'prob', 'them_cal', 'date_3', 'num_in_3'])

# mask = pickle.load(open('important1.p', 'rb'))
# input_vars = pickle.load(open('input_vars.p', 'rb'))
# input_vars = np.array(input_vars)[mask]

# input_vars = np.array([u'condtn', u'round', u'samerace', u'attr_o', u'sinc_o', u'fun_o', u'amb_o',
#                        u'like_o', u'prob_o', u'met_o', u'date', u'go_out', u'museums', u'art',
#                        u'amb3_1', u'attr', u'fun', u'amb', u'like', u'prob'])

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

