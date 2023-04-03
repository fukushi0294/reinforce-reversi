import numpy as np

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.n = 0
        self.q = 0.0

    def select_child(self, c=1.4):
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            score = child.q / child.n + c * np.sqrt(np.log(self.n) / child.n)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def rollout(self, env):
        state = self.state
        done = False
        while not done:
            action = np.random.choice(env.legal_actions(state))
            state, reward, done, _ = env.step(state, action)
        return reward

class MCTS:
    def __init__(self) -> None:
        pass