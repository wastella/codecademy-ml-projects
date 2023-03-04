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

data = df[["age", "capital-gain", "capital-loss", "hours-per-week", "sex"]]

train_data, test_data, train_labels, test_labels = train_test_split(data, income_data, randomstate=1)

classifier = RandomForestClassifier(random_state=1)