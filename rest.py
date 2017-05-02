import cPickle as pickle
import numpy as np

mask = pickle.load(open('important1match2.p', 'rb'))
# print pickle.load(open('important2.p', 'rb'))

input_vars = pickle.load(open('input_vars.p', 'rb'))

print mask
print np.array(input_vars)[mask]