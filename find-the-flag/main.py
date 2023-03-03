import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pd.read_csv("flags.csv", header=0)

labels = df["Landmass"]
data = df[["Red", "Green", "Blue", "Gold", "White", "Black", "Orange"]]