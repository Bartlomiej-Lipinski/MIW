import numpy as np  # Importuje bibliotekę do pracy na tablicach i macierzach
import matplotlib.pyplot as plt  # Importuje bibliotekę do tworzenia wykresów

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iterations=50):
        """
        Inicjalizacja perceptronu.
        
        Parametry:
        - learning_rate: Współczynnik uczenia (domyślnie 0.1).
        - n_iterations: Liczba iteracji treningowych (domyślnie 50).
        """
        self.learning_rate = learning_rate  # Ustawia współczynnik uczenia
        self.n_iterations = n_iterations  # Ustawia liczbę iteracji

    def train(self, X, y):
        """
        Uczenie perceptronu na podstawie danych treningowych.
        
        Parametry:
        - X: Macierz cech danych treningowych.
        - y: Wektor etykiet klas danych treningowych.
        """
        self.weights = np.zeros(X.shape[1] + 1)  # Inicjalizuje wagi
        for _ in range(self.n_iterations):  # Pętla ucząca
            for xi, target in zip(X, y):  # Pętla po danych treningowych
                prediction = self.predict(xi)  # Przewiduje klasę
                error = target - prediction  # Oblicza błąd
                self.weights[1:] += self.learning_rate * error * xi  # Aktualizuje wagi cech
                self.weights[0] += self.learning_rate * error  # Aktualizuje bias

    def predict(self, X):
        """
        Przewidywanie etykiety klasowej dla danych wejściowych.
        
        Parametry:
        - X: Dane wejściowe.
        
        Zwraca:
        - Wektor etykiet klasowych dla danych wejściowych.
        """
        activation = np.dot(X, self.weights[1:]) + self.weights[0]  # Oblicza aktywację
        return np.where(activation >= 0, 1, -1)  # Zwraca etykiety klas

    def accuracy(self, X, y):
        """
        Obliczenie dokładności klasyfikacji na podstawie danych wejściowych i prawdziwych etykiet.
        
        Parametry:
        - X: Dane wejściowe.
        - y: Prawdziwe etykiety klasowe.
        
        Zwraca:
        - Dokładność klasyfikacji jako ułamek.
        """
        predictions = self.predict(X)  # Przewiduje etykiety klas
        correct = np.sum(predictions == y)  # Liczy poprawne przewidywania
        total = len(y)  # Liczy łączną liczbę próbek
        return correct / total  # Zwraca dokładność klasyfikacji

# Generowanie losowych danych
# np.random.seed(0)
X_train = np.vstack([
    np.random.normal(loc=[4, 4], scale=[1, 1], size=(50, 2)),
    np.random.normal(loc=[2, 2], scale=[1, 1], size=(50, 2))
])  # Tworzy dane treningowe z dwóch różnych rozkładów normalnych
y_train = np.array([1] * 50 + [-1] * 50)  # Tworzy etykiety klas

# Podział danych na zbiór treningowy i testowy
indices = np.random.permutation(len(X_train))  # Losowo permutuje indeksy
split = int(0.8 * len(X_train))  # Określa punkt podziału
X_train, X_test = X_train[indices[:split]], X_train[indices[split:]]  # Dzieli dane treningowe i testowe
y_train, y_test = y_train[indices[:split]], y_train[indices[split:]]  # Dzieli etykiety klas na treningowe i testowe

# Przygotowanie danych do walidacji
n_iterations_list = list(range(10, 91, 10))  # Tworzy listę liczby iteracji do sprawdzenia
accuracy_list = []  # Inicjalizuje listę na dokładności

# Walidacja i wizualizacja granicy decyzyjnej
plt.figure(figsize=(10, 10))  # Tworzy nowy wykres
for i, n_iterations in enumerate(n_iterations_list, 1):  # Pętla po liczbie iteracji
    plt.subplot(3, 3, i)  # Tworzy subplot
    perceptron = Perceptron(n_iterations=n_iterations)  # Inicjalizuje perceptron z określoną liczbą iteracji
    perceptron.train(X_train, y_train)  # Trenuje perceptron na danych treningowych
    accuracy = perceptron.accuracy(X_test, y_test)  # Oblicza dokładność na danych testowych
    accuracy_list.append(accuracy)  # Dodaje dokładność do listy

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)  # Wyświetla dane treningowe
    plt.xlabel('Długość płatka')  # Dodaje opis osi x
    plt.ylabel('Szerokość płatka')  # Dodaje opis osi y
    plt.title(f'Liczba iteracji: {n_iterations}\nDokładność: {accuracy:.2f}')  # Dodaje tytuł wykresu

    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1  # Określa zakres osi x
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1  # Określa zakres osi y
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))  # Tworzy siatkę punktów
    Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])  # Przewiduje etykiety klas dla każdego punktu siatki
    Z = Z.reshape(xx.shape)  # Zmienia kształt przewidywań
    plt.contourf(xx, yy, Z, alpha=0.4)  # Wyświetla granicę decyzyjną

plt.tight_layout()  # Dopasowuje układ