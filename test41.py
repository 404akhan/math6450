import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import cPickle as pickle

df = pd.read_csv("./speed_code.csv", encoding="ISO-8859-1")

input_vars = pickle.load(open('input_vars_removed.p', 'rb'))
predict_col = 'dec_o'

# input_vars = input_vars[:10]

attribute_num = len(input_vars)
print 'attribute_num', attribute_num, 'predict', predict_col

xs = np.zeros((8378, attribute_num))
ys = np.zeros((8378, 1))

for i in range(attribute_num):
    xs[:, i] = df[input_vars[i]]
ys[:, 0] = df[predict_col]

xs[np.isnan(xs)] = 0.
ys = np.reshape(ys, len(ys))

# Create the RFE object and compute a cross-validated score.
clf = LogisticRegression()
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(5),
              scoring='accuracy')
selector = rfecv.fit(xs, ys)

print("Optimal number of features : %d" % rfecv.n_features_)
print selector.support_
print selector.ranking_

pickle.dump(selector.support_, open(predict_col+'important1removed.p', 'wb'))
pickle.dump(selector.ranking_, open(predict_col+'important2removed.p', 'wb'))
pickle.dump(rfecv.grid_scores_, open(predict_col+'grid_scores_removed.p', 'wb'))

plt.figure()
plt.xlabel(predict_col+": Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()