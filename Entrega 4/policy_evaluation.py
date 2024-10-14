import numpy as np


class PolicyEvaluation:
    def __init__(self, terminal_states, grid_size = (4, 4)):
        
        self.rows = grid_size[0]
        self.columns = grid_size[1]
        self.terminal_states = terminal_states   
        self.actions_list = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.v_current = np.zeros((self.rows, self.columns)) 

    def step(self, state, action):

        row, col = state
        shift_row, shift_col = action
        new_row = min(max(row+shift_row, 0), self.rows - 1)
        new_col = min(max(col+shift_col, 0), self.columns - 1)
        new_state = (new_row, new_col)

        return new_state

    def run_policy_evaluation(self, theshold = 0.01):
        V_new = self.v_current.copy()
        
        reward = -1 
        while True:
            delta = 0
            for row in range(self.rows):
                for col in range(self.columns):
                    state = (row, col)
                    if state in self.terminal_states:
                        continue
                    

                    new_value = 0
                    for action in self.actions_list:
                        next_state = self.step(state, action)
                        new_value += 0.25 * (reward + self.v_current[next_state[0], next_state[1]])
                    
                    V_new[row, col] = new_value
                    

                    delta = max(delta, abs(self.v_current[row, col] - V_new[row, col]))
            #print(V_new)
            self.v_current = V_new.copy()
            if delta <= theshold: 
                break
            
        return V_new

    def reset_vmap(self):
        self.v_current = np.zeros((self.rows, self.columns)) 


if __name__ == "__main__":
    print("Luís Fernando de Oliveira")
    grid_size = (4, 4)
    theshold = 0.0001
    terminal_states = [(0, 0), (3, 3)]
    p_ev = PolicyEvaluation(terminal_states=terminal_states, grid_size=grid_size)
    V_pi = p_ev.run_policy_evaluation(theshold)

    for row in V_pi:
        print(" | ".join(f"{v:6.2f}" for v in row))
