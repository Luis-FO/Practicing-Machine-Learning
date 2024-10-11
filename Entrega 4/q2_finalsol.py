import numpy as np
import matplotlib.pyplot as plt

def value_iteration(ph, theta=1e-20, max_capital=100):
    value_functions = []
    V_old = np.zeros(max_capital + 1) 
    V_old[100] = 1.0  
    V_new = V_old.copy()
    while True:
        delta = 0
        value_functions.append(V_old)
        for s in range(1, max_capital):
            action_returns = []
            for stake_action in range(1, min(s, max_capital - s) + 1):
                win_state = s + stake_action
                lose_state = s - stake_action
                action_return = ph * V_old[win_state] + (1 - ph) *V_old[lose_state]
                action_returns.append(action_return)

            V_new[s] = max(action_returns)
            delta = max(delta, abs(V_old[s] - V_new[s]))
        V_old = V_new.copy()
        
        # Critério de parada
        if delta < theta:
            
            break

    policy = np.zeros(max_capital + 1) 
    for s in range(1, max_capital):
        action_returns = []
        actions = range(1, min(s, max_capital - s) + 1)
        for stake in actions:
            win_state = s + stake
            lose_state = s - stake
            action_return = ph * V_old[win_state] + (1 - ph) * V_old[lose_state]
            action_returns.append(action_return)

        best_action = np.argmax(np.round(action_returns, 5))
        policy[s] = actions[best_action]

    return value_functions, policy

def plot_results(V, policy, ph):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    for v in V:
        ax1.plot(v)
    ax1.set_title(f"Value functions (p_h = {ph})")
    ax1.set_xlabel('Capital')
    ax1.set_ylabel('Value Estimates')

    # Gráfico da política final
    ax2.plot(policy, marker='o')
    ax2.set_title(f"Final Policy (p_h = {ph})")
    ax2.set_xlabel('Capital')
    ax2.set_ylabel('Final Policy (Stake)')

    plt.tight_layout()
    plt.show()

ph_values = [0.45, 0.25, 1]
for ph in ph_values:
    V, policy = value_iteration(ph)
    plot_results(V, policy, ph)
