Let's implement the policy evaluation algorithm for a 4x4 grid world based on the example described in Sutton and Barto's "Reinforcement Learning: An Introduction." In this example, we have a grid where an agent can move around and receive rewards based on its actions.

Here's a basic outline of the 4x4 grid world:

 - The agent starts in one of the grid cells.
 - The agent can move in four directions: up, down, left, and right.
 - Some cells may have terminal states with specific rewards.

## Policy Evaluation Implementation
We'll define the following components:

- Grid World: A 4x4 grid where the agent can take actions.
- Policy: A fixed policy that specifies which action to take in each state.
- Value Function: An array to hold the value of each state.

Here's how the implementation looks in Python:


## Explanation:
- Transition Function: Defines how the agent moves in the grid world based on its current state and action.
- Policy: A simple deterministic policy where the agent moves right if possible; otherwise, it moves down or up.
- Policy Evaluation Function: Iteratively updates the value function 
𝑉
V until convergence, using the Bellman equation to calculate expected rewards based on the defined policy.

## Output:

After running the code, you'll see the estimated value function for each state in the grid. The terminal states will have a value of 0, and other states will have values reflecting the expected future rewards.


## Print

To print the value function as a table (matrix) with estimated values of the states (𝑉𝜋) after convergence, we can modify the code slightly to format the output better. We'll ensure that the stopping condition is set to 
Δ ≤ 0.01
Δ≤0.01 for convergence.

Here's the updated implementation:


## Explanation of Changes:
- Theta Adjustment: The convergence threshold is now set to 0.01.
- Formatted Printing: The output of the value function is formatted as a matrix, where each value is printed with two decimal places for clarity.

## Output:
When you run the code, it will display the value function in a neat matrix format. Here's an example of what the output might look like:

Value Function (V_pi):
  -1.00 |  -1.00 |  -1.00 |  -1.00
  -1.00 |  -1.00 |  -1.00 |  -1.00
  -1.00 |  -1.00 |  -1.00 |  -1.00
  -1.00 |  -1.00 |  -1.00 |   0.00

The terminal states (0,0) and (3,3) will have specific values, while other states will reflect the expected future rewards based on the defined policy. Adjust the REWARD constant as needed to reflect the desired rewards for your specific scenario.


# Q2

Let's implement the solution for the Gambler's Problem using Value Iteration as described in Sutton and Barto's "Reinforcement Learning: An Introduction."

## Gambler's Problem Overview

In the Gambler's Problem, a gambler can bet a certain amount of their current fortune on a coin flip, which has a 50% chance of winning or losing. The goal is to maximize the expected reward, which in this case is reaching a certain amount of money (the goal state).

## Problem Setup

- The states represent the amount of money the gambler has, from 0 to 
N, where N is the goal.
- The gambler can choose how much to bet, ranging from 1 to the minimum of K (current fortune) or the remaining amount needed to reach the goal.
- The reward is defined as reaching the goal (winning) or going broke (losing).

## Implementation Steps
- 1:  Define the environment and the transition probabilities.
- 2: Implement the value iteration algorithm.
- 3: Plot the value function and policy.

Here’s how we can implement this in Python:

## Explanation of the Code:
- Value Initialization: The value function V is initialized for states 0 to N. The value of the goal state N is set to 1 (indicating success).
- Value Iteration Function: This function iteratively updates the value function using the Bellman equation until convergence.
- Optimal Policy Extraction: The optimal bet amount for each state is extracted by evaluating the expected values for all possible bets and selecting the one that maximizes the expected value.
- Plotting: The final value function and the optimal policy are plotted.

## Output:
- The first plot shows the value function V(k) for each state k, similar to Figure 4.3 in the book.
- The second plot illustrates the optimal betting policy for each state, showing the amount the gambler should bet based on their current fortune.

## Visual Comparison:
- After running the code, you can compare the plots with the ones in the book. The value function should resemble the one shown in the book, and the policy should indicate the optimal bet amounts based on the gambler's current state.

Feel free to adjust any parameters or the plotting style as needed!