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


class SequentialTrainingRunner:
    def __init__(self) -> None:
        self.episode = 10000
        self.trainee = QLearningAgent(1)
        self.opponent = RandomAgent(-1)

    def run(self):
        env = GameBoard()
        loss_history = []
        save_interval = 500
        trainee_win = 0
        win_history = []

        for i in range(self.episode):
            state = env.reset()
            total_loss, cnt = 0, 0
            done = False
            while not done:
                next_state, reward, action = self.one_step(env, state, i)
                if not next_state:
                    state = state.skip()
                    continue
                done = next_state.is_done()
                loss = self.trainee.train(
                    env, state, action, next_state, reward)
                total_loss += loss if np.isscalar(loss) else loss[0]
                cnt += 1
                state = next_state

            loss_avg = total_loss / cnt
            loss_history.append(loss_avg)

            if state.is_win(self.trainee.color):
                trainee_win += 1

            if (i+1) % 100 == 0:
                logger.info(
                    f"{i+1} game has done. Trainee won {trainee_win} times in recent 100 games")
                recent_loss_avg = np.mean(loss_history[i-99:i])
                logger.info(f"Recent loss average: {recent_loss_avg:.6f}")
                win_history.append(trainee_win)
                trainee_win = 0

            if (i+1) % save_interval == 0:
                print(f"{i+1} game has done. Save gradients as checkpoint")
                self.trainee.qnet.save()

    def batch_run(self):
        env = GameBoard()
        loss_history = []
        save_interval = 500
        win_history = []
        batch_size = 100

        for i in range(self.episode):
            total_loss, cnt = 0, 0
            trainee_win = 0
            states = [env.reset() for _ in range(batch_size)]
            while len(states) != 0:
                next_states = []
                current_board = []
                actions = []
                rewards = []
                next_boards = []
                next_actions = []
                for state in states:
                    next_state, reward, action = self.one_step(env, state, i)
                    if next_state:
                        if not next_state.is_done():
                            next_states.append(next_state)
                        else:
                            trainee_win += 1 if next_state.is_win(
                                self.trainee.color) else 0

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
            win_history.append(trainee_win)

            logger.info(
                f"{i+1}00 game has done. Trainee won {trainee_win} times in recent 100 games")
            logger.info(f"Recent loss average: {loss_avg:.6f}")
            win_history.append(trainee_win)

    def one_step(self, env: GameBoard, state: State, game_cnt: int):
        try:
            trainee_action = self.trainee.get_action(env, state)
            # trainee_action = self.trainee.get_action_by_boltzmann(
            #     env, state, game_cnt)
            next_state, reward, _ = env.step(
                state, trainee_action, self.trainee.color)
        except NotFoundLegalActionException:
            return None, 0, -1

        if next_state.is_done():
            return next_state, reward, trainee_action

        try:
            action = self.opponent.get_action(env, next_state)
            next_state, reward, _ = env.step(
                next_state, action, self.opponent.color)
            return next_state, -reward, trainee_action
        except NotFoundLegalActionException:
            return next_state.skip(), 0, trainee_action


if __name__ == '__main__':
    SequentialTrainingRunner().batch_run()
