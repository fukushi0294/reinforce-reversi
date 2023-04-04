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

    # evaluate board state with rate of win
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
        self.n += 1
        if state.is_done():
            if state.is_win(state.color):
                self.q += 1
                return 1
            else:
                return 0

        actions = self.state.legal_actions()
        if len(actions):
            for action in actions:
                next_state, _, _ = env.step(state, action)
                child = Node(next_state, self)
                self.children.append(child)

        child: Node = np.random.choice(self.children)
        result = child.rollout()
        if result == 0:
            self.q += 1
            return 1
        else:
            return 0


class MCTS:
    def __init__(self, state: State, env: GameBoard) -> None:
        self.root = Node(state)
        self.env = env

    def simulate(self, env: GameBoard):
        pass

    # input state is owned by opponent
    def proceed(self, state):
        current_node: Node = None
        for child in self.root.children:
            if child.state == state:
                current_node == child

        next_root = current_node.select_child()
        self.root = next_root
        return next_root.state
