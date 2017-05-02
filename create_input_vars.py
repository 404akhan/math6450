import cPickle as pickle

input_vars = pickle.load(open('input_vars_miss1000.p', 'rb'))

print input_vars
print len(input_vars)

input_vars_remove = []

for k in input_vars:
    if k != 'prob' and k != 'prob_o' and k != 'id' and k != 'iid' and k != 'idg' and k != 'them_cal':
        input_vars_remove.append(k)

print input_vars_remove
print len(input_vars_remove)

pickle.dump(input_vars_remove, open('input_vars_removed.p', 'wb'))

