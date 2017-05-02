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

df = pd.read_csv("./speed_code.csv", encoding="ISO-8859-1")
predict_col = 'dec_o'
rem1 = np.array(pickle.load(open(predict_col+'important1removed.p', 'rb')))
input_vars = np.array(pickle.load(open('input_vars_removed.p', 'rb')))
input_vars = np.array(input_vars[rem1])
input_vars = np.concatenate((input_vars, ['prob', 'prob_o']))

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
    xs, ys, test_size=0.33, random_state=40)

second_row = []

def apply_model(clf):
    global second_row
    clf.fit(X_train, y_train)

    predict = clf.predict(X_test)
    print predict
    error = np.mean(np.square(predict - y_test))
    print '%.2f%%' % ((1 - error) * 100)
    second_row.append('%.2f%%' % ((1 - error) * 100))
    return predict

models = {}
models['LogisticRegression'] = LogisticRegression()
models['SVC'] = SVC()
models['LinearDiscriminantAnalysis'] = LinearDiscriminantAnalysis()
models['QuadraticDiscriminantAnalysis'] = QuadraticDiscriminantAnalysis()
models['RandomForestClassifier'] = RandomForestClassifier()
models['ExtraTreesClassifier'] = ExtraTreesClassifier()
models['DecisionTreeClassifier'] = DecisionTreeClassifier()


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
print '\nmodel: Ensemble'
print '%.2f%%' % ((1 - error) * 100)

first_row.append('Ensemble')
second_row.append('%.2f%%' % ((1 - error) * 100))

# pickle.dump([first_row, second_row], open(predict_col+'performance.p', 'wb'))

print first_row
print second_row
#
# with open('results/' + predict_col+'-prediction-accuracy.csv', 'wb') as csvfile:
#     spamwriter = csv.writer(csvfile, delimiter=',',
#                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     for i in range(len(first_row)):
#         spamwriter.writerow([first_row[i], second_row[i]])
