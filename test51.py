# t-sne
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
predict_col = 'dec'
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
    xs, ys, test_size=0.33, random_state=43)

arr = np.zeros(8378)
arr[:] = df['like']
arr[:] = df['prob_o']
arr[np.isnan(arr)] = 6.
print arr
print ys
print np.corrcoef(arr, ys)
# plt.plot(arr, ys, '.')
# plt.show()
sys.exit()

def apply_model(clf):
    clf.fit(X_train, y_train)

    predict = clf.predict(X_test)
    # print predict
    error = np.mean(np.square(predict - y_test))
    print '%.2f%%' % ((1 - error) * 100)
    return predict

models = {}
models['Logistic Regression'] = LogisticRegression()
models['SVC'] = SVC()
models['LDA'] = LinearDiscriminantAnalysis()
models['QDA'] = QuadraticDiscriminantAnalysis()
# models['RandomForestClassifier'] = RandomForestClassifier()
# models['ExtraTreesClassifier'] = ExtraTreesClassifier()
# models['DecisionTreeClassifier'] = DecisionTreeClassifier()

# model = TSNE(n_components=2, random_state=0)
# np.set_printoptions(suppress=True)
# arr = model.fit_transform(X_test)

pca = PCA(n_components=2)
pca.fit(X_test)
Components = pca.components_
arr = np.matmul(X_test, Components.transpose())

dict = {'dec': 'DEC', 'dec_o': 'Dec_o', 'match': 'Match'}

for k, v in models.iteritems():
    predict = apply_model(v)
    print predict

    plt.title('Predictions of ' + dict[predict_col] + ' using '+ k)
    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.plot(arr[predict!=y_test, 0], arr[predict!=y_test, 1], 'r.', label='incorrect')
    plt.plot(arr[predict==y_test, 0], arr[predict==y_test, 1], 'g.', label='correct')
    plt.legend()
    plt.show()

# plt.plot(arr[y_test==0, 0], arr[y_test==0, 1], 'r.')
# plt.plot(arr[y_test==1, 0], arr[y_test==1, 1], 'g.')
# plt.show()

sys.exit()

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

print first_row
print second_row

