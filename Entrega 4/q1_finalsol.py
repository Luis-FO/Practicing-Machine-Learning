import numpy as np

# Constantes
GRID_SIZE = 4
NUM_ACTIONS = 4  # Acima, Abaixo, Esquerda, Direita
TERMINAL_STATES = [(0, 0), (3, 3)]  # Estados terminais
REWARD = -1  # Recompensa para cada movimento até alcançar o estado terminal

# Define o modelo de transição para o gridworld
def transition(state, action):
    if state in TERMINAL_STATES:
        return state  # Nenhuma transição a partir dos estados terminais
    
    row, col = state
    if action == 0:  # Acima
        new_state = (max(row - 1, 0), col)
    elif action == 1:  # Abaixo
        new_state = (min(row + 1, GRID_SIZE - 1), col)
    elif action == 2:  # Esquerda
        new_state = (row, max(col - 1, 0))
    elif action == 3:  # Direita
        new_state = (row, min(col + 1, GRID_SIZE - 1))
    
    return new_state

# Avaliação de política com política equiprovável
def policy_evaluation():
    V_old = np.zeros((GRID_SIZE, GRID_SIZE))  # Função de valor inicializada com zeros
    V_curr = np.zeros((GRID_SIZE, GRID_SIZE))
    while True:
        delta = 0
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                state = (row, col)
                if state in TERMINAL_STATES:
                    continue
                
                # Calcula o valor esperado sobre todas as ações equiprováveis
                new_value = 0
                for action in range(NUM_ACTIONS):
                    next_state = transition(state, action)
                    new_value += 0.25 * (REWARD + V_old[next_state[0], next_state[1]])  # Política equiprovável
                
                V_curr[row, col] = new_value
                
                # Calcula a diferença máxima (delta) entre o valor anterior e o novo valor
                delta = max(delta, abs(V_old[row, col] - V_curr[row, col]))
        print(V_curr)
        input()
        if delta < 0.01:  # Condição de convergência
            break
        V_old = V_curr.copy()
    
    return V_curr

# Avaliar a política
V_equiprobable = policy_evaluation()

# Imprimir a função de valor como uma matriz
print("Função de Valor (V_pi) com Política Equiprovável:")
for row in V_equiprobable:
    print(" | ".join(f"{v:6.2f}" for v in row))
