import cPickle as pickle
import numpy as np
import pandas as pd

df = pd.read_csv("./speed_code.csv", encoding="ISO-8859-1")

mask = pickle.load(open('important1match2.p', 'rb'))
input_vars = pickle.load(open('input_vars.p', 'rb'))
input_vars = np.array(input_vars)[mask]

for i in input_vars:
    print '\'' + i + '\',',
