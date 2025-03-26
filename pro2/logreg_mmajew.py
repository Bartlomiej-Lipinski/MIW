# %%
import numpy as np  # Importuje bibliotekę NumPy do pracy na tablicach i macierzach
import matplotlib.pyplot as plt  # Importuje bibliotekę Matplotlib do tworzenia wykresów

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate  # Ustawia współczynnik uczenia
        self.n_iterations = n_iterations  # Ustawia liczbę iteracji

    def train(self, X, y):
        n_samples, n_features = X.shape  # Pobiera liczbę próbek i cech
        self.weights = np.zeros(n_features)  # Inicjalizuje wagi na zero
        self.bias = 0  # Inicjalizuje obciążenie na zero

        for _ in range(self.n_iterations):  # Pętla ucząca
            linear_model = np.dot(X, self.weights) + self.bias  # Oblicza model liniowy
            y_predicted = self.sigmoid(linear_model)  # Przewiduje wartości za pomocą funkcji sigmoidalnej

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))  # Oblicza gradient wag
            db = (1 / n_samples) * np.sum(y_predicted - y)  # Oblicza gradient obciążenia

            self.weights -= self.learning_rate * dw  # Aktualizuje wagi
            self.bias -= self.learning_rate * db  # Aktualizuje obciążenie

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias  # Oblicza model liniowy
        y_predicted = self.sigmoid(linear_model)  # Przewiduje wartości za pomocą funkcji sigmoidalnej
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]  # Dokonuje klasyfikacji na podstawie progowania
        return y_predicted_cls  # Zwraca przewidywane klasy
    
    def predict_probability(self, X):
        linear_model = np.dot(X, self.weights) + self.bias  # Oblicza model liniowy
        y_predicted = self.sigmoid(linear_model)  # Przewiduje wartości za pomocą funkcji sigmoidalnej
        return y_predicted  # Zwraca przewidywane klasy
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))  # Implementacja funkcji sigmoidalnej

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


def main():
    # Generowanie losowych danych
    np.random.seed(0)
    # Tworzenie danych treningowych z dwóch różnych rozkładów normalnych
    X_train = np.vstack([
        np.random.normal(loc=[4, 4], scale=[1, 1], size=(50, 2)),
        np.random.normal(loc=[2, 2], scale=[1, 1], size=(50, 2))
    ])
    y_train = np.array([1] * 50 + [0] * 50)  # Tworzy etykiety klas

    # Podział danych na zbiór treningowy i testowy
    indices = np.random.permutation(len(X_train))  # Losowo permutuje indeksy
    split = int(0.8 * len(X_train))  # Określa punkt podziału
    X_train, X_test = X_train[indices[:split]], X_train[indices[split:]]  # Dzieli dane treningowe i testowe
    y_train, y_test = y_train[indices[:split]], y_train[indices[split:]]  # Dzieli etykiety klas na treningowe i testowe

    # Przygotowanie danych do walidacji
    n_iterations = 10000
    logisticregression = LogisticRegression(n_iterations=n_iterations)  # Inicjalizuje LogisticRegression z określoną liczbą iteracji
    logisticregression.train(X_train, y_train)  # Trenuje logisticregression na danych treningowych
    accuracy = logisticregression.accuracy(X_test, y_test)  # Oblicza dokładność na danych testowych
    
    # Wyświetlanie danych treningowych i granicy decyzyjnej
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    plt.xlabel('Długość płatka')
    plt.ylabel('Szerokość płatka')
    plt.title(f'Liczba iteracji: {n_iterations}\nDokładność: {accuracy:.2f}')

    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1  # Określa zakres osi x
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1  # Określa zakres osi y
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))  # Tworzy siatkę punktów
    Z = np.array(logisticregression.predict(np.c_[xx.ravel(), yy.ravel()]))  # Przewiduje etykiety klas dla każdego punktu siatki
    print(type(Z))
    Z = Z.reshape(xx.shape)  # Zmienia kształt przewidywań
    plt.contourf(xx, yy, Z, alpha=0.4)  # Wyświetla granicę decyzyjną
    
    plt.show()
    

if __name__ == '__main__':
    main()
# %%
