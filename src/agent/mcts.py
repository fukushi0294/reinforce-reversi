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

    def greedy_probs(self, epsilon: float):
        best_child = self.select_child()
        base_prob = epsilon / len(self.children)
        select_probs = {action: base_prob for action in self.children}
        select_probs[best_child] += (1 - epsilon)
        return select_probs

    def rollout(self, env: GameBoard, epsilon: float):
        edge = self
        # go down to edge when children node is all searched
        while len(edge.children) > 0 and all(lambda x: x.n == 0, edge.children):
            edge = np.random.choice(
                edge.children, edge.greedy_probs(epsilon).values())

        if edge.state.is_done():
            return edge

        if len(edge.children) == 0:
            state = edge.state
            actions = edge.state.legal_actions()
            for action in actions:
                next_state, _, _ = env.step(state, action)
                child = Node(next_state, self)
                edge.children.append(child)
        untried = filter(lambda x: x.n == 0, edge.children)
        edge = np.random.choice(untried)
        return edge

    def backpropagete(self, color):
        node = self
        while node is not None:
            node.n += 1
            if node.state.is_win(color):
                node.q += 1
            node = node.parent


class MCTS:
    def __init__(self, state: State, env: GameBoard) -> None:
        self.root = Node(state)
        self.env = env
        self.epsilon = 0.1
        self.episode = 1024

    def simulate(self):
        for _ in range(self.episode):
            edge = self.root.rollout(self.env, self.epsilon)
            edge.backpropagete(self.root.state.color)

    # input state is owned by opponent
    def proceed(self, state):
        current_node: Node = None
        for child in self.root.children:
            if child.state == state:
                current_node == child

        next_root = current_node.select_child()
        self.root = next_root
        return next_root.state
