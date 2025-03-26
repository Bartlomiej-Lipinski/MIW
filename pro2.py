from perceptron_class import Perceptron as Perceptron
from logistic_regression import LogisticRegression, calculate_accuracy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(42)
cluster_centers = [[1, 1], [7, 7], [1, 7], [7, 1]]
std_dev = 0.8
X = np.vstack([np.random.normal(center, std_dev, (50, 2)) for center in cluster_centers])
y = np.hstack([[i] * 50 for i in range(4)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
plt.figure(figsize=(8, 6))
for i, color in zip(range(4), ["red", "blue", "green", "purple"]):
    plt.scatter(X_train[y_train == i][:, 0], X_train[y_train == i][:, 1], label=f'Klasa {i}', alpha=0.6, color=color)

plt.legend()
plt.title("Wygenerowane dane - zbiór treningowy")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
#######################################################################################################################
Perceptron = Perceptron(learning_rate=0.1, n_iterations=500)
Perceptron.fit(X_train, y_train)
y_pred = Perceptron.predict(X_test)
accuracy = calculate_accuracy(y_test, y_pred)
#######################################################################################################################
# LogisticRegression = LogisticRegression(learning_rate=0.1, n_iterations=500)
# LogisticRegression.fit(X_train, y_train)
# y_pred = LogisticRegression.predict(X_test)
# # accuracy = calculate_accuracy(y_test, y_pred)

