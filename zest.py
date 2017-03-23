import numpy as np
import pandas as pd
import random
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
import cPickle as pickle

xs_fields = np.array(['attr', 'sinc', 'intel', 'fun', 'amb', 'shar', 'like', 'prob',
                        'attr_o', 'sinc_o', 'intel_o', 'fun_o', 'amb_o', 'shar_o', 'like_o', 'prob_o', 'samerace',
                        "sports","tvsports","exercise","dining","museums","art","hiking","gaming",
                        "clubbing","reading","tv","theater","movies","concerts","music","shopping","yoga",
                        "sports_o","tvsports_o","exercise_o","dining_o","museums_o","art_o","hiking_o","gaming_o",
                        "clubbing_o","reading_o","tv_o","theater_o","movies_o","concerts_o","music_o","shopping_o","yoga_o",
                      ])

get_diff =  np.array(["sports","tvsports","exercise","dining","museums","art","hiking","gaming",
                        "clubbing","reading","tv","theater","movies","concerts","music","shopping","yoga"])

xs = pickle.load(open('new_xs.p'))

new_xs = np.zeros((xs.shape[0], xs.shape[1]-len(get_diff)))

print xs
print xs.shape[1] == len(xs_fields)

for i in range(xs.shape[1] - len(get_diff)):
    if i < xs.shape[1] - 2*len(get_diff):
        new_xs[:, i] = xs[:, i]
    else:
        # new_xs[:, i] = np.exp(-np.square(xs[:, i] - xs[:, i + len(get_diff)])/2)
        new_xs[:, i] = 1./( np.square(xs[:, i] - xs[:, i + len(get_diff)]) +1e-5)

print new_xs.shape
print new_xs

pickle.dump(new_xs, open('lala.p', 'wb'))


