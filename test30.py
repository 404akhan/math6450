import cPickle as pickle
import numpy as np
import pandas as pd

df = pd.read_csv("./speed_code.csv", encoding="ISO-8859-1")

mask = pickle.load(open('important1match.p', 'rb'))
input_vars = pickle.load(open('input_vars.p', 'rb'))
input_vars = np.array(input_vars)[mask]

attribute_num = len(input_vars)
print 'attribute_num', attribute_num

xs = np.zeros((8378, attribute_num))
ys = np.zeros((8378))

for i in range(attribute_num):
    xs[:, i] = df[input_vars[i]]
ys[:] = df['match']

# xs[np.isnan(xs)] = 6.

count = 0
arr_vars = []

for i in range(attribute_num):
    miss_num = np.sum(np.isnan(xs[:, i]))
    tot_num = np.sum(1-np.isnan(xs[:, i])) + miss_num
    print i, input_vars[i], '%d / %d' % (miss_num, tot_num)
    # if miss_num > 500:
    #     print input_vars[i]
    # else:
    #     count += 1
    #     arr_vars.append((input_vars[i]))
#
# print count
# print '\n', arr_vars
