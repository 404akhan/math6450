import cPickle as pickle
import numpy as np

rem1 = pickle.load(open('important1match_miss1000.p', 'rb'))

input_vars = pickle.load(open('input_vars_miss1000.p', 'rb'))

print input_vars[rem1]

rem2 = np.array(input_vars[rem1])

pickle.dump(rem2, open('final-match-vars.p', 'wb'))
