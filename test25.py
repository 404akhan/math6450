import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

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

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()