if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from agent.q_learning_agent import QLearningAgent
from agent.random_agent import RandomAgent
from common.state import State
from env.gameboard import GameBoard
from common.exception import NotFoundLegalActionException
import numpy as np
import logging
import pandas as pd


logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class SelfPlayTrainingRunner:
    def __init__(self) -> None:
        self.episode = 10000
        self.trainee = QLearningAgent(1)
        self.opponent = RandomAgent(-1)

    def run(self):
        env = GameBoard()
        loss_history = []
        save_interval = 10
        batch_size = 100

        for i in range(self.episode):
            total_loss, cnt = 0, 0
            states = [env.reset() for _ in range(batch_size)]
            while len(states) != 0:
                next_states = []
                current_board = []
                actions = []
                rewards = []
                next_boards = []
                next_actions = []
                for state in states:
                    next_state, reward, action = self.self_play(env, state)
                    if not next_state.is_done():
                        next_states.append(next_state)

                    if action != -1:
                        current_board.append(state.board)
                        actions.append(action)
                        rewards.append(reward)
                        next_boards.append(next_state.board)
                        next_action = np.array(env.get_actions(
                            next_state, self.trainee.color))
                        next_actions.append(next_action)

                if len(current_board) == 0 or len(next_boards) == 0:
                    break

                df = pd.DataFrame({
                    'current_board': current_board,
                    'action': actions,
                    'next_board': next_boards,
                    'next_actions': next_actions,
                    'reward': rewards
                })
                loss = self.trainee.batch_train(df)
                total_loss += loss if np.isscalar(loss) else np.mean(loss)
                cnt += 1
                states = next_states

            loss_avg = total_loss / cnt
            loss_history.append(loss_avg)

            logger.info(
                f"{i+1}00 game has done. Recent loss average: {loss_avg:.6f}")

            if (i+1) % save_interval == 0:
                logger.info(
                    f"{i+1}00 game has done. Save gradients as checkpoint")
                self.trainee.qnet.save()

                trainee_win = 0
                states = [env.reset() for _ in range(batch_size)]
                while len(states) != 0:
                    next_states = []
                    current_board = []
                    next_boards = []
                    for state in states:
                        next_state = self.one_step(env, state)
                        if not next_state.is_done():
                            next_states.append(next_state)
                        else:
                            trainee_win += 1 if next_state.is_win(
                                self.trainee.color) else 0
                    cnt += 1
                    states = next_states

                logger.info(
                    f"100 games are simulated. Trainee won {trainee_win} times in recent 100 games")

    def self_play(self, env: GameBoard, state: State):
        try:
            trainee_action = self.trainee.get_greedy_action(env, state)
            next_state, reward, _ = env.step(
                state, trainee_action, state.next_turn)
            return next_state, reward, trainee_action
        except NotFoundLegalActionException:
            return state.skip(), 0, -1

    def one_step(self, env: GameBoard, state: State):
        try:
            trainee_action = self.trainee.get_action(env, state)
            next_state, _, _ = env.step(
                state, trainee_action, self.trainee.color)
        except NotFoundLegalActionException:
            next_state = state.skip()

        if next_state.is_done():
            return next_state

        try:
            action = self.opponent.get_action(env, next_state)
            next_state, _, _ = env.step(
                next_state, action, self.opponent.color)
            return next_state
        except NotFoundLegalActionException:
            return next_state.skip()


if __name__ == '__main__':
    SelfPlayTrainingRunner().run()
