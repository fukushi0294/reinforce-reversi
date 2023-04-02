if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from collections import defaultdict
from env.gameboard import GameBoard
from agent.simple_agent import SimpleAgent
from typing import List, Tuple, Dict


class RandomAgent(SimpleAgent):
    def __init__(self, player_color: int = 1):
        self.color = player_color

    def reset(self):
        pass

    def get_action(self, state: Tuple[int, List[Tuple[int, int]]]):
        if state[0] != self.color:
            return None
        if len(state[1]) == 0:
            return None
        p = 1/len(state[1])
        return {s[0] * 8 + s[1]: p for s in state[1]}

    def step(self, observation, reward, done):
        pass

# predict one step
def eval_onestep(agent: RandomAgent, V, env: GameBoard, gamma=0.9):
    state = env.states()
    action_probs = agent.get_action(state)
    new_V = 0

    for action, action_probs in action_probs.items():
        next_state, r, _ = env.step(action)
        new_V += action_probs * (r + gamma * V[next_state])

    V[state] = new_V
    return V

def policy_eval(agent: RandomAgent, V, env: GameBoard, gamma, threshold=0.001):
    while True:
        old_V = V.copy()
        V = eval_onestep(agent, V, env, gamma)
        env.reset()

        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t
        
        if delta < threshold:
            break
    
    return V


if __name__ == '__main__':
    env = GameBoard()
    gamma = 0.9

    agent = RandomAgent()
    V = defaultdict(lambda: 0)
    V = policy_eval(agent, V, env, gamma)
    # env.render_v(V)
