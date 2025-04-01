import numpy as np  # Importowanie biblioteki NumPy do operacji na macierzach i wektorach
import matplotlib.pyplot as plt  # Importowanie biblioteki Matplotlib do tworzenia wykresów
from perceptron_mmajew import Perceptron  # Importowanie zaimplementowanej klasy Perceptron

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
split = int(0.8 * size_of_data)  # Określenie rozmiaru zbioru treningowego na podstawie proporcji
X_train = np.array([X[_, :split] for _ in range(len(X))])  # Podział danych na zbiór treningowy
X_test = np.array([X[_, split:] for _ in range(len(X))])  # Podział danych na zbiór testowy

# Inicjalizacja listy do przechowywania modeli perceptronów i trenowanie ich
models = []  # Inicjalizacja listy do przechowywania modeli
X_train_all = np.concatenate([X_train[i] for i in range(4)], axis=0)
y_train_all = np.concatenate([np.full(X_train[i].shape[0], i) for i in range(4)], axis=0)
for class_idx in range(4):
    y_binary = np.where(y_train_all == class_idx, 1, -1)
    model = Perceptron(0.1,1000)
    model.train(X_train_all, y_binary)
    models.append(model)
# Inicjalizacja modeli one-versus-one
ovo_models = {}
for i in range(4):
    for j in range(i + 1, 4):
        indices = np.where((y_train_all == i) | (y_train_all == j))[0]
        X_pair = X_train_all[indices]
        y_pair = np.where(y_train_all[indices] == i, 1, -1)
        model = Perceptron(0.1,500)
        model.train(X_pair, y_pair)
        ovo_models[(i, j)] = model

# Przewidywanie klas dla danych testowych i obliczanie dokładności
X_test_all = np.concatenate([X_test[i] for i in range(4)], axis=0)
y_test_all = np.concatenate([np.full(X_test[i].shape[0], i) for i in range(4)], axis=0)
activations = np.zeros((X_test_all.shape[0], 4))
for idx, model in enumerate(models):
    activations[:, idx] = X_test_all.dot(np.array(model.weights[1:])) + model.weights[0]
preds_ovr = np.argmax(activations, axis=1)
accuracy_ovr = np.mean(preds_ovr == y_test_all)
# Przewidywanie klas dla danych testowych w modelu one-versus-one
preds_ovo = []
for x in X_test_all:
    votes = np.zeros(4)
    for (i, j), model in ovo_models.items():
        activation = np.dot(np.array(model.weights[1:]), x) + model.weights[0]
        if activation >= 0:
            votes[i] += 1
        else:
            votes[j] += 1
    preds_ovo.append(np.argmax(votes))
preds_ovo = np.array(preds_ovo)
accuracy_ovo = np.mean(preds_ovo == y_test_all)

print("Dokładność one-versus-rest:", accuracy_ovr)
print("Dokładność one-versus-one:", accuracy_ovo)
# Wyświetlanie punktów danych treningowych i testowych
colors = ['red', 'green', 'blue', 'magenta']  # Lista kolorów dla różnych klas

fig, axs = plt.subplots(1, 2, figsize=(14, 7))

# Subplot dla one-versus-rest
for _ in range(4):
    axs[0].scatter(X_train[_][:, 0], X_train[_][:, 1], label=f'Class {_}', color=colors[_], marker='o')  # Wyświetlenie punktów treningowych
    axs[0].scatter(X_test[_][:, 0], X_test[_][:, 1], color=colors[_], marker='x')  # Wyświetlenie punktów testowych

# Rysowanie granic decyzyjnych modeli perceptronów
min_x1 = np.min(X[:,:,0])
max_x1 = np.max(X[:,:,0])
min_x2 = np.min(X[:,:,1])
max_x2 = np.max(X[:,:,1])

for _ in range(4):
    [c, a, b] = models[_].weights  # Współczynniki prostej decyzyjnej
    # Zakres dla zmiennej x
    x_range = np.array([min_x1, max_x1])
    # Obliczenie wartości zmiennej y na podstawie równania prostej
    y_range = (-a * x_range - c) / b
    # Tworzenie wykresu
    axs[0].plot(x_range, y_range, color=colors[_])  # Rysowanie granic decyzyjnych dla każdej klasy

axs[0].set_xlim(min_x1, max_x1)  # Ustawienie limitów dla osi x
axs[0].set_ylim(min_x2, max_x2)  # Ustawienie limitów dla osi y
axs[0].set_title('One-Versus-Rest')
axs[0].legend()

# Subplot dla one-versus-one
for _ in range(4):
    axs[1].scatter(X_train[_][:, 0], X_train[_][:, 1], label=f'Class {_}', color=colors[_], marker='o')  # Wyświetlenie punktów treningowych
    axs[1].scatter(X_test[_][:, 0], X_test[_][:, 1], color=colors[_], marker='x')  # Wyświetlenie punktów testowych

min_x1 = np.min(X[:,:,0])
max_x1 = np.max(X[:,:,0])
min_x2 = np.min(X[:,:,1])
max_x2 = np.max(X[:,:,1])

for _ in range(4):
    model = ovo_models[_].weights  # Pobranie wag perceptronu
    [c, a, b] = model.weights  # Pobranie wag perceptronu

    x_range = np.array([min_x1, max_x1])  # Zakres wartości X
    y_range = (-a * x_range - c) / b  # Obliczenie wartości Y

    axs[1].plot(x_range, y_range, color=colors[_])  # Rysowanie granic decyzyjnych dla pary klas

axs[1].set_xlim(min_x1, max_x1)  # Ustawienie limitów dla osi x
axs[1].set_ylim(min_x2, max_x2)  # Ustawienie limitów dla osi y
axs[1].set_title('One-Versus-One')
axs[1].legend()

# Zapisanie i wyświetlenie wykresu
plt.savefig('06b perceptron multiclass_new.png')
plt.show()  # Wyświetlenie wszystkich punktów danych oraz granic decyzyjnych