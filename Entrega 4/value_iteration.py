import numpy as np
import matplotlib.pyplot as plt
import random

class ValueIteration:
    def __init__(self, ph = 0.4, max_capital=100):
        
        self.max_capital = max_capital
        self.ph = ph
        self.v_current = np.zeros(max_capital + 1) 
        self.v_current[100] = 1.0  

    def run_value_iteration(self, theshold=1e-20):
        value_functions = []
        V_new = self.v_current.copy()

        while True:
            delta = 0
            value_functions.append(self.v_current)
            for s in range(1, self.max_capital):
                action_returns = []
                for stake_action in range(1, min(s, self.max_capital - s) + 1):
                    win_state = s + stake_action
                    lose_state = s - stake_action
                    action_return = self.ph * self.v_current[win_state] + (1 - self.ph) *self.v_current[lose_state]
                    action_returns.append(action_return)

                V_new[s] = max(action_returns)
                delta = max(delta, abs(self.v_current[s] - V_new[s]))
            self.v_current = V_new.copy()
            
            if delta < theshold:
                break
        return value_functions

    def getPolicy(self):
        policy = np.zeros(self.max_capital + 1) 
        for s in range(1, self.max_capital):
            action_returns = []
            actions = range(1, min(s, self.max_capital - s) + 1)
            for stake in actions:
                win_state = s + stake
                lose_state = s - stake
                action_return = self.ph * self.v_current[win_state] + (1 - self.ph) * self.v_current[lose_state]
                action_returns.append(action_return)


            action_returns = [np.round(value, 10) for value in action_returns]
            best_value = np.max(action_returns)

            ############ Essa parte serve apenas para visualizar as ações possíveis em cada estado #########
            best_actions = [idx for idx,value in enumerate(action_returns) if value==best_value]
            print(f"Current State {s}")
            print(f"Best action: {[a+1 for a in best_actions]}")
            ###############################################################################################
            
            # Para escolha aleatória, descomente a "LINHA 1" e comente a "LINHA 2"
            #best_action = random.choice(best_actions) #---- LINHA 1
            best_action = np.argmax(action_returns)    #---- LINHA 2
            
            policy[s] = actions[best_action]

        return policy
    
    def reset_vmap(self):
        self.v_current = np.zeros(self.max_capital + 1) 
        self.v_current[100] = 1.0
    
    def setPh(self, new_ph):
        self.ph = new_ph

def plot_results(V, policy, ph):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    for v in V:
        ax1.plot(v)
    ax1.set_title(f"Value functions (p_h = {ph})")
    ax1.set_xlabel('Capital')
    ax1.set_ylabel('Value Estimates')

    ax2.plot(policy, '-o')
    ax2.set_title(f"Final Policy (p_h = {ph})")
    ax2.set_xlabel('Capital')
    ax2.set_ylabel('Final Policy (Stake)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Luís Fernando de Oliveira")
    val_iter = ValueIteration(ph = 0.4, max_capital=100)
    ph_values = [0.4, 0.25, 0.55]
    theshold = 1e-20
    for ph in ph_values:
        val_iter.setPh(ph)
        V = val_iter.run_value_iteration(theshold=theshold)
        policy = val_iter.getPolicy()
        plot_results(V, policy, ph)
        val_iter.reset_vmap()
