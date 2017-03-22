import numpy as np
import pandas as pd
import random
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

df = pd.read_csv("./speed_code.csv", encoding="ISO-8859-1")
input_vars = np.array(df.keys())
print np.array(df.keys())