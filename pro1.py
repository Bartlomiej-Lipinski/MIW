import random
import numpy as np
import matplotlib.pyplot as plt

# Opis: Program symuluje grę w "Kamień, Papier, Nożyce" pomiędzy graczem a komputerem,
#       aktualizując macierz przejść na podstawie wyników i uczenia się w czasie rzeczywistym.

# Inicjalizacja stanu gotówki gracza
cash = 0
cash_history = [cash]

##### KOMPUTER #####
# Definicja ruchów/taktyki komputera
states_computer = ["Paper", "Rock", "Scissors"]
transition_matrix_computer = {
    "Paper": {"Paper": 2 / 3, "Rock": 1 / 3, "Scissors": 0 / 3},
    "Rock": {"Paper": 0 / 3, "Rock": 2 / 3, "Scissors": 1 / 3},
    "Scissors": {"Paper": 2 / 3, "Rock": 0 / 3, "Scissors": 1 / 3}
}

# Przekształcenie macierzy przejść transition_matrix_computer do postaci tablicy numpy
transition_matrix_np = {
    state: np.array(list(transition_matrix_computer[state].values()))
    for state in states_computer
}


def choose_move(previous_move, transition_matrix):
    return np.random.choice(states_computer, p=transition_matrix[previous_move])


##### GRACZ #####
# Wersja 1: Obliczanie wektora stacjonarnego macierzy przejść transition_matrix_computer
stationary_vector = np.linalg.matrix_power(np.array([
    [2 / 3, 1 / 3, 0],
    [0, 2 / 3, 1 / 3],
    [2 / 3, 0, 1 / 3]
]), 1000)[0]
player_strategy_v1 = {state: stationary_vector[i] for i, state in enumerate(states_computer)}


def choose_move_v1():
    return np.random.choice(states_computer, p=list(player_strategy_v1.values()))


# Wersja 2: Inicjalizacja macierzy przejść gracza
player_transition_matrix = {state: {s: 1 / 3 for s in states_computer} for state in states_computer}


def update_player_transition(previous_move, current_move):
    for move in states_computer:
        player_transition_matrix[previous_move][move] *= 0.9
    player_transition_matrix[previous_move][current_move] += 0.1


def choose_move_v2(previous_move):
    probabilities = list(player_transition_matrix[previous_move].values())
    return np.random.choice(states_computer, p=probabilities)


def determine_winner(player, computer):
    if player == computer:
        return 0  # remis
    elif (player == "Rock" and computer == "Scissors") or \
            (player == "Scissors" and computer == "Paper") or \
            (player == "Paper" and computer == "Rock"):
        return 1  # gracz wygrywa
    else:
        return -1  # komputer wygrywa


# Wybór strategii gracza (1 lub 2)
strategy_version = 1  # Można zmienić na 1, jeśli chcemy używać wersji 1

# Główna pętla gry
player_last_move = random.choice(states_computer)
computer_last_move = random.choice(states_computer)

for _ in range(1000):
    computer_move = choose_move(computer_last_move, transition_matrix_np)

    if strategy_version == 1:
        player_move = choose_move_v1()
    else:
        player_move = choose_move_v2(player_last_move)
        update_player_transition(player_last_move, player_move)

    result = determine_winner(player_move, computer_move)
    cash += result
    cash_history.append(cash)

    player_last_move = player_move
    computer_last_move = computer_move

# Wykres zmiany stanu gotówki w każdej kolejnej grze
plt.plot(range(1001), cash_history)
plt.xlabel('Numer Gry')
plt.ylabel('Stan Gotówki')
plt.title('Zmiana Stanu Gotówki w Grze "Kamień, Papier, Nożyce"')
plt.show()