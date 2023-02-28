from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

breast_cancer_data = load_breast_cancer()

train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size=0.2, random_state=100)