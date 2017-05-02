import numpy as np
import pandas as pd
import random
import statsmodels.api as sm
import sys
from pylab import pcolor, show, colorbar, xticks, yticks
import cPickle as pickle

df = pd.read_csv("./speed_code.csv", encoding="ISO-8859-1")

# print df[0]
# sys.exit()
input_vars = np.array(['attr', 'sinc', 'intel', 'fun', 'amb', 'shar', 'like', 'prob',
                       'attr_o', 'sinc_o', 'intel_o', 'fun_o', 'amb_o', 'shar_o', 'like_o', 'prob_o',
                       'samerace',
                        "sports","tvsports","exercise","dining","museums","art","hiking","gaming",
                        "clubbing","reading","tv","theater","movies","concerts","music","shopping","yoga",])

xs_fields = np.array(['attr', 'sinc', 'intel', 'fun', 'amb', 'shar', 'like', 'prob',
                        'attr_o', 'sinc_o', 'intel_o', 'fun_o', 'amb_o', 'shar_o', 'like_o', 'prob_o', 'samerace',
                        "sports","tvsports","exercise","dining","museums","art","hiking","gaming",
                        "clubbing","reading","tv","theater","movies","concerts","music","shopping","yoga",
                        "sports_o","tvsports_o","exercise_o","dining_o","museums_o","art_o","hiking_o","gaming_o",
                        "clubbing_o","reading_o","tv_o","theater_o","movies_o","concerts_o","music_o","shopping_o","yoga_o",
                      ])

pickle.dump(xs_fields, open('xs_fields.p', 'wb'))
sys.exit()

interests = ["sports","tvsports","exercise","dining","museums","art","hiking","gaming",
        "clubbing","reading","tv","theater","movies","concerts","music","shopping","yoga"]

print np.array(df).shape
print xs_fields.shape

def get_vals(i):
    for interest in interests:
        print df[interest][i]

# get_vals(0)

def get_ids_o(i):
    arr = []
    for index in range(len(df['iid'])):
        iid = df['iid'][index]
        if iid == i:
            arr.append(df['pid'][index])
    return arr

# print get_ids_o(1)

def get_attributes_o(i, attrs):
    arr_gl = []
    for atr in attrs:
        arr = []
        arr.append(df[atr][i])
    return arr

def array_iid_props():
    # arr[iid] = [atrs]
    arr = np.zeros((555, 17))
    for i in range(df.shape[0]):
        for j in range(len(interests)):
            arr[ df['iid'][i] ][j] = df[interests[j]][i]
    return arr

iid_props = array_iid_props()
print iid_props

def get_new_xs():
    xs = np.zeros(( df.shape[0], len(xs_fields) ))
    for i in range(df.shape[0]):
        for j in range(len(xs_fields)):
            if j < len(input_vars):
                xs[i, j] = df[xs_fields[j]][i]
            else:
                # print type(df['pid'][i])
                # print df['pid'][i]
                tmp1 = int(df['pid'][i]) if not np.isnan(df['pid'][i]) else 0
                if np.isnan(df['pid'][i]):
                    print 'None', df['pid'][i], i
                # print tmp1
                xs[i, j:] = iid_props[ tmp1 ]
                break
    return xs

new_xs = get_new_xs()

print new_xs.shape
print new_xs

pickle.dump(new_xs, open('new_xs.p', 'wb'))
print 'dumped'

# todo
# 1
# need a new table like this sports, exercise, sports_o, exercise_o
# done in 2 iterations create arr[iid] = [sports, exercise]
# second iter for each iid append arr[pid]

# 2
# read given vars again

# 3
# read kernels

# 4
# linear regr on some vars and your own