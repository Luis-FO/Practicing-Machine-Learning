import numpy as np

# Constants
GRID_SIZE = 4
NUM_ACTIONS = 4  # Up, Down, Left, Right
TERMINAL_STATES = [(0, 0), (3, 3)]  # Define terminal states (with rewards)
REWARD = -1  # Reward for each step until reaching terminal states

# Define the transition model for the grid world
def transition(state, action):
    if state in TERMINAL_STATES:
        return state  # No transition from terminal states
    
    row, col = state
    if action == 0:  # Up
        new_state = (max(row - 1, 0), col)
    elif action == 1:  # Down
        new_state = (min(row + 1, GRID_SIZE - 1), col)
    elif action == 2:  # Left
        new_state = (row, max(col - 1, 0))
    elif action == 3:  # Right
        new_state = (row, min(col + 1, GRID_SIZE - 1))
    
    return new_state

# Policy: deterministic policy for this example
def policy(state):
    # Simple policy: always move right unless at the edge
    if state[1] < GRID_SIZE - 1:
        return 3  # Move right
    elif state[0] < GRID_SIZE - 1:
        return 1  # Move down
    return 0  # Move up (to prevent going out of bounds)

# Policy evaluation function
def policy_evaluation(policy, theta=0.01):
    V = np.zeros((GRID_SIZE, GRID_SIZE))  # Initialize value function
    while True:
        delta = 0
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                state = (row, col)
                if state in TERMINAL_STATES:
                    continue
                
                v = V[row, col]
                action = policy(state)  # Get action from policy
                new_state = transition(state, action)
                
                # Update value function
                V[row, col] = REWARD + V[new_state[0], new_state[1]]  # Bellman equation
                
                # Calculate the maximum change in value
                delta = max(delta, abs(v - V[row, col]))
        
        if delta < theta:  # Check convergence
            break
    
    return V

# Run the policy evaluation
value_function = policy_evaluation(policy)

# Print the value function as a matrix
print("Value Function (V_pi):")
for row in value_function:
    print(" | ".join(f"{v:6.2f}" for v in row))
