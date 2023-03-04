def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("income.csv", header=0, delimiter = ", ")

income_data = df[["income"]]