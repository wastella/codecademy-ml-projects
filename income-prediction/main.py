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

df["sex-int"] = df["sex"].apply(lambda row: 0 if row == "Male" else 1)
df["country-int"] = df["native-country"].apply(lambda row: 0 if row == "United-States" else 1)

data = df[["age", "capital-gain", "capital-loss", "hours-per-week", "sex-int", "country-int"]]

train_data, test_data, train_labels, test_labels = train_test_split(data, income_data, random_state=1)

classifier = RandomForestClassifier(random_state=1)

classifier.fit(train_data, train_labels)

print(classifier.score(train_data, train_labels))