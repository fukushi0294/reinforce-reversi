if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from collections import defaultdict
from env.gameboard import GameBoard
from agent.simple_agent import SimpleAgent
from common.state import State
from common.exception import NotFoundLegalActionException
import numpy as np
from agent.mcts import MCTS, Node


class MonteCarloAgent(SimpleAgent):
    def __init__(self, player_color: int = 1):
        self.color = player_color
        self.mcts: MCTS = None

    def reset(self):
        pass

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
