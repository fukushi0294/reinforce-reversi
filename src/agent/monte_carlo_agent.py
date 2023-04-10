if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from typing import List
from env.gameboard import GameBoard
from agent.simple_agent import SimpleAgent
from common.state import State
from common.exception import NotFoundLegalActionException
import numpy as np


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
        while len(edge.children) > 0 and all(x.n != 0 for x in edge.children):
            select_probs = edge.greedy_probs(epsilon)
            edge = np.random.choice(
                list(select_probs.keys()), p=list(select_probs.values()))

        if edge.state.is_done():
            return edge

        if len(edge.children) == 0:
            state = edge.state
            actions = env.get_actions(edge.state, state.next_turn)
            if len(actions) == 0:
                next_state = state.skip()
                child = Node(next_state, edge)
                edge.children.append(child)
            else:
                for action in actions:
                    next_state, _, _ = env.step(state, action, state.next_turn)
                    child = Node(next_state, edge)
                    edge.children.append(child)
        untried = list(filter(lambda x: x.n == 0, edge.children))
        return np.random.choice(untried)

    def backpropagate(self, color):
        node = self
        q = 1 if node.state.is_win(color) else 0
        while node is not None:
            node.n += 1
            node.q += q
            node = node.parent


class MCTS:
    def __init__(self, env: GameBoard, state: State, color: int) -> None:
        self.env = env
        self.root = Node(state)
        self.player_color = color
        self.epsilon = 0.1
        self.episode = 4000

    def simulate(self):
        for _ in range(self.episode):
            edge = self.root.rollout(self.env, self.epsilon)
            edge.backpropagate(self.player_color)

    def get_action(self):
        node = self.root.select_child()
        if node is None:
            raise NotFoundLegalActionException("No legal action")
        return node.state.memory[-1]

    def proceed(self, last_action: int):
        current_node: Node = None
        for child in self.root.children:
            if child.state.memory[-1] == last_action:
                current_node = child
        if current_node is None:
            print("illegal state")
        self.root = current_node


class MonteCarloAgent(SimpleAgent):
    def __init__(self, player_color: int = 1):
        self.color = player_color
        self.mcts: MCTS = None

    def reset(self):
        self.mcts = None

    def get_action(self, env: GameBoard, state: State):
        if self.mcts is None:
            self.mcts = MCTS(env, state, self.color)
        else:
            self.mcts.proceed(state.memory[-1])
        self.mcts.simulate()
        action = self.mcts.get_action()
        self.mcts.proceed(action)
        return action

    def step(self, observation, reward, done):
        pass
