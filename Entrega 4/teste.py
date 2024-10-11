import numpy as np
from random import random, choice


class MDP:
    def __init__(self, l, c) -> None:
        self.l = l
        self.c = c

        self.v_pi = np.zeros((l, c))
        self.terminal_states = [(0, 0), (3, 3)]

    def transition(self, state, action):
        if state in self.terminal_states:
            return state
        
        l, c = state

        if action==0: # UP
            new_state = (max(l-1, 0), c)
        elif action==1: # R
            new_state = (l, min(c+1, self.c-1))
        elif action==2: #D
            new_state = (min(l+1, self.l-1), c)
        elif action==3: #L
            new_state = (l, max(c-1, 0))
        return new_state
    
    def policy_evaluation(self):
        while True:
            delta = 0
            for l in range(self.l):
                for c in range(self.c):
                    state = (l, c)
                    if state in self.terminal_states:
                        continue
                    
                    v = self.v_pi[state]
                    action = self.policy()
                    new_state = self.transition(state, action)
                    self.v_pi[l, c] = -1 + self.v_pi[new_state]
                    print(self.v_pi)
                    delta = max(delta, abs(v-self.v_pi[state]))
            if delta<1e-10:
                break
                    #input()


    def policy(self):
        actions = [0, 1, 2, 3]
        return choice(actions)

# def Gt(r):
#     pass

# def main():
#     gamma = 0.1
#     r = 0
#     gt = r
#     for i in range(10):
#         v = random()
#         if v>=0.5:
#             r = 1
#         else:
#             r = -1

#         gt += (r*pow(gamma, i))
#         print(gt)

if __name__ == "__main__":
    obj = MDP(4, 4)
    state = (1, 1)
    obj.policy_evaluation()