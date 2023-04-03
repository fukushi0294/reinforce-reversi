import numpy as np
from env.gameboard import GameBoard
from common.state import State
from typing import List


class Node:
    def __init__(self, state: State, parent=None):
        self.state: State = state
        self.parent: Node = parent
        self.children: List[Node] = []
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

    def rollout(self, env: GameBoard):
        state = self.state
        done = False
        while not done:
            actions = self.state.legal_actions()
            action = np.random.choice(actions)
            state, reward, done = env.step(state, action)
        return reward
    
    def simulate(self, env: GameBoard):
      node = self
      done = False
      while not done:
          if not node.children:
              action = np.random.choice(node.state.legal_actions())
              next_state, reward, done = env.step(node.state, action)
              child_node = Node(next_state, node)
              node.children.append(child_node)
              return reward
          else:
              node = node.select_child()
              _, reward, done = env.step(node.state, node.state.last_action)
      return reward

class MCTS:
    def __init__(self, state: State, env: GameBoard) -> None:
        self.max_depth = 5
        self.root = Node(state)
        self.env = env

    def extend(self):
        self.extend_recusive(self.root, 0)
    
    def extend_recusive(self, node: Node, depth: int):
        if depth == self.max_depth:
            return
        
        actions = node.state.legal_actions()
        for action in actions:
            next_state, reward, done = self.env.step(node, action)
            if done:
                return
            node.children.append(Node(next_state, node))
        
        depth =+ 1
        for child in node.children:
            self.extend_recusive(child, depth)
    
    def proceed(self):
        pass
    
    def evaluate(self):
        pass
    