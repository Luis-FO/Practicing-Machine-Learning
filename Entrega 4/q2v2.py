import numpy as np
import matplotlib.pyplot as plt

def value_iteration(ph, theta=1e-10, max_capital=100):
    # Valores iniciais
    V = np.zeros(max_capital + 1)
    V[100] = 1.0  # Se o jogador atinge 100, a recompensa é 1

    policy = np.zeros(max_capital + 1)

    while True:
        delta = 0
        for s in range(1, max_capital):
            print(f"State: {s}")
            old_v = V[s]
            action_returns = []

            # O jogador pode apostar de 1 até o mínimo entre s ou 100 - s
            for stake in range(1, min(s, max_capital - s) + 1):
                win_state = s + stake
                lose_state = s - stake
                action_return = ph * V[win_state] + (1 - ph) * V[lose_state]
                print("Stake:", stake)
                print(action_return)
                #input()
                action_returns.append(action_return)

            # Atualiza o valor da política e o valor de V
            V[s] = max(action_returns)
            delta = max(delta, abs(old_v - V[s]))

        # Critério de parada
        if delta < theta:
            break

    # Deriva a política ótima a partir da função valor final
    for s in range(1, max_capital):
        action_returns = []
        for stake in range(1, min(s, max_capital - s) + 1):
            win_state = s + stake
            lose_state = s - stake
            action_return = ph * V[win_state] + (1 - ph) * V[lose_state]
            action_returns.append(action_return)
        best_action = np.argmax(action_returns) + 1
        policy[s] = best_action

    return V, policy

def plot_results(V, policy, ph):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Gráfico da função de valor
    ax1.plot(V)
    ax1.set_title(f"Value Estimates (p_h = {ph})")
    ax1.set_xlabel('Capital')
    ax1.set_ylabel('Value')

    # Gráfico da política final
    ax2.plot(policy)
    ax2.set_title(f"Final Policy (p_h = {ph})")
    ax2.set_xlabel('Capital')
    ax2.set_ylabel('Final Policy (Stake)')

    plt.tight_layout()
    plt.show()

# Exemplo de execução com ph = 0.25 e ph = 0.55
ph_values = [0.45, 0.55]
for ph in ph_values:
    V, policy = value_iteration(ph)
    plot_results(V, policy, ph)
