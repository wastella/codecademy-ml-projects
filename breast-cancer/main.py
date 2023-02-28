from sklearn.datasets import load_breast_cancer

breast_cancer_data = load_breast_cancer()

print(breast_cancer_data.data[0])

print(breast_cancer_data.feature_names)