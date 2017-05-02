import cPickle as pickle
import numpy as np
import csv

model_dec, dec = pickle.load(open('decperformance.p', 'rb'))
model_dec_o, dec_o = pickle.load(open('dec_o'+'performance.p', 'rb'))
model_match, match = pickle.load(open('match'+'performance.p', 'rb'))

print model_dec, model_dec_o, model_match
print dec

with open('results-removed-90/prediction-accuracy.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['model_name', 'dec', 'dec_o', 'match'])
    for i in range(len(model_dec)):
        spamwriter.writerow([model_dec[i], dec[i], dec_o[i], match[i]])

