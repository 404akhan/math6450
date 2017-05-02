import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import csv

# rem1 = pickle.load(open('important1match_miss1000.p', 'rb'))
#
# input_vars = pickle.load(open('input_vars_miss1000.p', 'rb'))
#
# print input_vars[rem1]
#
# rem2 = np.array(input_vars[rem1])
#
# pickle.dump(rem2, open('final-match-vars.p', 'wb'))

dec = pickle.load(open('dec'+'grid_scores_removed.p', 'rb'))
dec_o = pickle.load(open('dec_o'+'grid_scores_removed.p', 'rb'))
match = pickle.load(open('match'+'grid_scores_removed.p', 'rb'))
first_row = range(1, len(dec_o) + 1)

with open('results-removed-90/recursive-selection.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['num-of-vars', 'dec', 'dec_o', 'match'])
    for i in range(len(first_row)):
        spamwriter.writerow([first_row[i], dec[i], dec_o[i], match[i]])
