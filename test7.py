import numpy as np
import pandas as pd
import random
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
import sys
from pylab import pcolor, show, colorbar, xticks, yticks
import matplotlib.pyplot as plt

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

xs[np.isnan(xs)] = 11.
ys[np.isnan(ys)] = 11.

plt.plot(xs[:, 8], ys[:, 0], '.')
plt.show()



sys.exit()
############
_ = 0
for i in range(attribute_num):
    print input_vars[i]
    times = np.zeros(11)
    for j in range(11):
        # print (xs[:,i] == j).astype(int) + (ys[:,0] == 1).astype(int)
        times[j] = np.sum((xs[:,i] == j).astype(int) + (ys[:,0] == 0).astype(int) == 2)
    plt.plot(times)
    plt.show()


