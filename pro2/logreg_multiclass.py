import numpy as np
import matplotlib.pyplot as plt
from logreg_mmajew import LogisticRegression

def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    softmax_values = exp_x / sum_exp_x
    return softmax_values

# Generowanie danych
np.random.seed(0)  # Ustawienie ziarna losowości dla powtarzalności wyników
size_of_data = 50  # Określenie liczby punktów danych w każdej klasie
X = np.array([
    np.random.normal(loc=[1, 1], scale=[1, 1], size=(size_of_data, 2)),  # Generowanie punktów dla pierwszej klasy
    np.random.normal(loc=[10, 10], scale=[1, 2], size=(size_of_data, 2)),  # Generowanie punktów dla drugiej klasy
    np.random.normal(loc=[1, 10], scale=[1, 2], size=(size_of_data, 2)),  # Generowanie punktów dla trzeciej klasy
    np.random.normal(loc=[10, 1], scale=[1, 1], size=(size_of_data, 2))  # Generowanie punktów dla czwartej klasy
])

# Podział danych na zbiór treningowy i testowy
split = int(0.8 * size_of_data)
X_train = np.array([X[_, :split] for _ in range(len(X))])
X_test = np.array([X[_, split:] for _ in range(len(X))])
#######################################################################################################################
# Inicjalizacja i trenowanie modeli regresji logistycznej
models = []  # Inicjalizacja listy do przechowywania modeli
for class_idx in range(4):
    X_train_all = np.concatenate([X_train[i] for i in range(4)], axis=0)
    y_train_all = np.concatenate([np.ones(split) if i == class_idx else np.zeros(split) for i in range(4)])
    LR = LogisticRegression(learning_rate=0.1, n_iterations=500)
    LR.train(X_train_all, y_train_all)
    models.append(LR)
true_labels = []
predicted_labels = []
# Dla każdej klasy przechodzimy po wszystkich punktach testowych
for class_idx in range(4):
    for sample in X_test[class_idx]:
        probs = np.array([model.predict_probability([sample])[0] for model in models])
        norm_probs = softmax(probs)
        pred_class = np.argmax(norm_probs)
        true_labels.append(class_idx)
        predicted_labels.append(pred_class)
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)
accuracy = np.mean(true_labels == predicted_labels)
print("Dokładność klasyfikacji: {:.2f}%".format(accuracy * 100))
########################################################################################################################
# Wyświetlanie punktów danych treningowych i testowych
colors = ['red', 'green', 'blue', 'magenta']
for _ in range(4):
    plt.scatter(X_train[_][:, 0], X_train[_][:, 1], label=f'Class {_}', color=colors[_], marker='o')
    plt.scatter(X_test[_][:, 0], X_test[_][:, 1], color=colors[_], marker='x')

min_x1 = np.min(X[:,:,0])
max_x1 = np.max(X[:,:,0])
min_x2 = np.min(X[:,:,1])
max_x2 = np.max(X[:,:,1])

for _ in range(4):
    [a, b] = models[_].weights
    c = models[_].bias
    # Zakres dla zmiennej x
    x_range = np.array([min_x1, max_x1])
    # Obliczenie wartości zmiennej y na podstawie równania prostej
    y_range = (-a * x_range - c) / b
    # Tworzenie wykresu
    plt.plot(x_range, y_range, color=colors[_])

# Ustawienie limitów dla osi x i y
plt.xlim(min_x1, max_x1)  # Ustawienie limitów dla osi x
plt.ylim(min_x2, max_x2)  # Ustawienie limitów dla osi y

# Wyświetlenie wykresu
plt.savefig('07b logistic regression multiclass_new.png')
plt.show()  # Wyświetlenie wykresu

# Obliczanie softmax dla środkowego punktu wykresu
x1_middle = (min_x1 + max_x1) / 2
x2_middle = (min_x2 + max_x2) / 2
y_prob = np.array([models[_].predict_probability([[x1_middle, x2_middle]]) for _ in range(len(X))]).flatten()
print(f"Prawdopodobieństwa przynależności {x1_middle, x2_middle} reg. log.: {y_prob}")
print(f"Wektor softmax {x1_middle, x2_middle}: {softmax(y_prob)}")

# Obliczanie softmax dla punktu (0,0) na wykresie
y_prob = np.array([models[_].predict_probability([[0, 0]]) for _ in range(len(X))]).flatten()
print(f"Prawdopodobieństwa przynależności {0, 0} reg. log.: {y_prob}")
print(f"Wektor softmax {0, 0}: {softmax(y_prob)}")