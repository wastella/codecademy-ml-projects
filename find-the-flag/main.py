import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pd.read_csv("flags.csv", header=0)

labels = df["Landmass"]
data = df[["Red", "Green", "Blue", "Gold", "White", "Black", "Orange"]]

train_data, test_data, train_labels, test_labels = train_test_split(data, labels)

tree = DecisionTreeClassifier(random_state=1)

tree.fit(train_data, train_labels)

