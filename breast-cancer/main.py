from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

breast_cancer_data = load_breast_cancer()

training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size=0.2, random_state=100)

classifier = KNeighborsClassifier(n_neighbors=3)

classifier.fit(training_data, training_labels)

score = classifier.score(validation_data, validation_labels)

accuracies = []

for i in range(1, 101):
    test_classifier = KNeighborsClassifier(n_neighbors=i)
    test_classifier.fit(training_data, training_labels)
    accuracies.append(test_classifier.score(validation_data, validation_labels))

k_list = range(1, 101)

plt.plot(k_list, accuracies)
plt.show()