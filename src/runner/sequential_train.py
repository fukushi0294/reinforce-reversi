if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from agent.q_learning_agent import QLearningAgent
from agent.random_agent import RandomAgent
from env.gameboard import GameBoard
from common.exception import NotFoundLegalActionException


class SequentialTrainingRunner:
    def __init__(self) -> None:
        self.episode = 10000
        self.trainee = QLearningAgent(1)
        self.opponent = RandomAgent(-1)

    def run(self):
        env = GameBoard()
        loss_history = []

        for _ in range(self.episode):
            state = env.reset()
            total_loss, cnt = 0, 0
            done = False
            while not done:
                try:
                    action = self.trainee.get_action(env, state)
                except NotFoundLegalActionException:
                    continue
                next_state, _, done = env.step(
                    state, action, self.trainee.color)
                if done:
                    loss = self.trainee.train(env, state)
                    break
                try:
                    action = self.opponent.get_action(env, next_state)
                except NotFoundLegalActionException:
                    loss = self.trainee.train(env, state)
                    continue
                next_state, reward, done = env.step(
                    state, action, self.opponent.color)
                loss = self.trainee.train(
                    env, state, next_state, reward, done)

                total_loss += loss
                cnt += 1
                state = next_state

            loss_avg = total_loss / cnt
            loss_history.append(loss_avg)


if __name__ == '__main__':
    SequentialTrainingRunner().run()
