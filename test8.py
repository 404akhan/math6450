import numpy as np
import pandas as pd
import random

df = pd.read_csv("./speed_code.csv", encoding="ISO-8859-1")
input_vars = np.array(['attr_o', 'sinc_o', 'intel_o', 'fun_o', 'amb_o', 'shar_o', 'like_o', 'prob_o'])

attribute_num = len(input_vars)
print 'attribute_num', attribute_num

xs = np.zeros((8378, attribute_num))
ys = np.zeros((8378, 1))

for i in range(attribute_num):
    xs[:, i] = df[input_vars[i]]
ys[:, 0] = df['dec_o']

xs[np.isnan(xs)] = 5.

random.seed(1339)
shuf_arr = range(0, 8378)
random.shuffle(shuf_arr)
train_size = int(8378 * 0.7)
lr = 0.1

xs_train = xs[shuf_arr[0:train_size], :]
xs_cross_val = xs[shuf_arr[train_size:], :]
ys_train = ys[shuf_arr[0:train_size], :]
ys_cross_val = ys[shuf_arr[train_size:], :]

xs_mean = np.mean(xs_train, axis=0)
xs_std = np.std(xs_train, axis=0)

xs_train = (xs_train - xs_mean) / xs_std
xs_cross_val = (xs_cross_val - xs_mean) / xs_std
