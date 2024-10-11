import numpy as np
import matplotlib.pyplot as plt

# Constants
N = 100  # Goal amount
p_win = 0.4  # Probability of winning
p_lose = 0.6  # Probability of losing

# Initialize the value function
V = np.zeros(N + 1)  # Value function for states 0 to N
V[N] = 1.0  # Value of reaching the goal

# Define the Value Iteration function
def value_iteration(V, theta=1e-6):
    while True:
        delta = 0
        for k in range(1, N):
            v = V[k]  # Old value
            # Calculate the maximum expected value for each action (bet)
            action_values = []
            for bet in range(1, min(k, N - k) + 1):
                win_value = p_win * V[k + bet] + p_lose * V[k - bet]
                action_values.append(win_value)
            V[k] = max(action_values)  # Update value function
            delta = max(delta, abs(v - V[k]))  # Check convergence
        if delta < theta:
            break
    return V

# Run Value Iteration
V = value_iteration(V)

# Extract the optimal policy
policy = np.zeros(N + 1)
for k in range(1, N):
    action_values = []
    for bet in range(1, min(k, N - k) + 1):
        win_value = p_win * V[k + bet] + p_lose * V[k - bet]
        action_values.append(win_value)
    optimal_bet = np.argmax(action_values) + 1  # +1 because bet starts from 1
    policy[k] = optimal_bet

# Plot the value function
plt.figure(figsize=(12, 6))
plt.plot(range(N + 1), V, label='Value Function (V)')
plt.xlabel('State (Fortune)')
plt.ylabel('Value')
plt.title("Value Function for the Gambler's Problem")
plt.grid()
plt.legend()
plt.xlim(0, N)
plt.ylim(0, 1)
plt.axhline(0, color='black', lw=0.8)
plt.axvline(0, color='black', lw=0.8)
plt.axvline(N, color='black', lw=0.8)
plt.show()

# Plot the policy
plt.figure(figsize=(12, 6))
plt.plot(range(N + 1), policy, label='Policy', marker='o')
plt.xlabel('State (Fortune)')
plt.ylabel('Optimal Bet Amount')
plt.title("Optimal Policy for the Gambler's Problem")
plt.grid()
plt.legend()
plt.xlim(0, N)
plt.ylim(0, N)
plt.axhline(0, color='black', lw=0.8)
plt.axvline(0, color='black', lw=0.8)
plt.axvline(N, color='black', lw=0.8)
plt.show()
