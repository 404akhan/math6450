import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

df = pd.read_csv("./speed_code.csv", encoding="ISO-8859-1")
input_vars = ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar', 'like', 'prob']

f = df.loc[df.gender == 1, :]
f_data = f.copy()
f_data = f.dropna(subset=input_vars)
f_model = sm.OLS(f_data.dec, sm.add_constant(f_data.loc[:, input_vars]))
f_results = f_model.fit()
print f_results.params
print f_results.pvalues

estimator = SVC(kernel= "linear", C=0.1)
selector = RFE(estimator,1)
cfl=selector.fit(f_data.loc[:, input_vars], f_data.dec)
ranking_f=cfl.ranking_
feature_ranking_f=zip(ranking_f, input_vars)
print list(feature_ranking_f)

