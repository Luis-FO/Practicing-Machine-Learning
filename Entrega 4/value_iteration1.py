import numpy as np

# Inicializa o array V com zeros, exceto no estado 100
V = np.zeros(101)
V[100] = 1  # Estado terminal onde o jogador ganha

# Definimos a probabilidade de ganhar
p = 0.2

# Função para calcular a recompensa esperada para uma ação 'a' a partir de um estado 's'
def backup_action(s, a, V, p):
    return p * V[s + a] + (1 - p) * V[s - a]

# Implementação do value iteration
def value_iteration(epsilon=1e-8):
    while True:
        delta = 0
        for s in range(1, 100):
            old_V = V[s]
            # Para o estado s, escolhemos a melhor ação possível (maximizar o valor esperado)
            V[s] = max(backup_action(s, a, V, p) for a in range(1, min(s, 100 - s) + 1))
            # Calcula a diferença máxima entre o valor antigo e o novo
            delta = max(delta, abs(old_V - V[s]))
        # Critério de parada
        if delta < epsilon:
            break
    return V

# Função para derivar a política a partir da função valor
def policy(s, epsilon=1e-10):
    best_value = -1
    best_action = None
    for a in range(1, min(s, 100 - s) + 1):
        this_value = backup_action(s, a, V, p)
        if this_value > best_value + epsilon:
            best_value = this_value
            best_action = a
    return best_action

# Executa o value iteration
V_final = value_iteration()

# Exemplo de como usar a função 'policy' para obter a melhor ação para um estado
state = 15
best_action = policy(state)
print(f"A melhor ação para o estado {state} é apostar {best_action}")
