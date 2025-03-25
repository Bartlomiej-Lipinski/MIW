from perceptron_class import Perceptron as Perceptron
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegression, calculate_accuracy

np.random.seed(42)
cluster_centers = [[2, 2], [6, 6], [2, 6], [6, 2]]
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
class MultiClassPerceptronOvR:
    def __init__(self, learning_rate=0.1, n_iterations=50):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.perceptrons = {}

    def train(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            y_binary = np.where(y == c, 1, -1)
            perceptron = Perceptron(learning_rate=self.learning_rate, n_iterations=self.n_iterations)
            perceptron.train(X, y_binary)
            self.perceptrons[c] = perceptron

    def predict(self, X):
        scores = np.array([perceptron.predict(X) for perceptron in self.perceptrons.values()])
        return np.argmax(scores, axis=0)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

ovr_perceptron = MultiClassPerceptronOvR(n_iterations=100)
ovr_perceptron.train(X_train, y_train)
accuracy_ovr = ovr_perceptron.accuracy(X_test, y_test)
print(f'Accuracy of OvR Perceptron: {accuracy_ovr * 100:.2f}%')
#######################################################################################################################
class MultiClassPerceptronOvO:
    def __init__(self, learning_rate=0.1, n_iterations=50):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.perceptrons = {}

    def train(self, X, y):
        self.classes = np.unique(y)
        self.pairwise_classifiers = list(combinations(self.classes, 2))  # Wszystkie pary klas

        for (class_a, class_b) in self.pairwise_classifiers:
            indices = np.where((y == class_a) | (y == class_b))
            X_binary, y_binary = X[indices], y[indices]
            y_binary = np.where(y_binary == class_a, 1, -1)
            perceptron = Perceptron(learning_rate=self.learning_rate, n_iterations=self.n_iterations)
            perceptron.train(X_binary, y_binary)
            self.perceptrons[(class_a, class_b)] = perceptron

    def predict(self, X):
        votes = {c: np.zeros(X.shape[0]) for c in self.classes}
        for (class_a, class_b), perceptron in self.perceptrons.items():
            predictions = perceptron.predict(X)
            for i, pred in enumerate(predictions):
                if pred == 1:
                    votes[class_a][i] += 1
                else:
                    votes[class_b][i] += 1

        final_predictions = np.array([max(votes, key=lambda c: votes[c][i]) for i in range(X.shape[0])])
        return final_predictions

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

# ovo_perceptron = MultiClassPerceptronOvO(n_iterations=100)
# ovo_perceptron.train(X_train, y_train)
# accuracy_ovo = ovo_perceptron.accuracy(X_test, y_test)
#######################################################################################################################

#######################################################################################################################
LogisticRegression = LogisticRegression(learning_rate=0.1, n_iterations=500)
LogisticRegression.fit(X_train, y_train)
y_pred = LogisticRegression.predict(X_test)
accuracy = calculate_accuracy(y_test, y_pred)

