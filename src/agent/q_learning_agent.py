from common.state import State
from agent.simple_agent import SimpleAgent
from env.gameboard import GameBoard
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from keras.optimizers import SGD


class QNet:
    def __init__(self) -> None:
        self.model = Sequential([
            Dense(512, activation='relu', input_shape=(64,)),
            Dense(512, activation='relu'),
            Dense(64, activation='relu')
        ])
        self.optimizer = SGD(learning_rate=0.001)

    def forward(self, state: State):
        X = state.board.flatten().reshape(1, -1)
        return self.model(X, training=True)

    def backward(self, tape: tf.GradientTape, loss):
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))


class QLearningAgent(SimpleAgent):
    def __init__(self, player_color: int = 1, agent: SimpleAgent = None):
        self.color = player_color
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.qnet = QNet()
        self.opponent = agent

    def reset(self):
        pass

    def get_action(self, env: GameBoard, state: State):
        actions = env.get_actions(state, self.color)
        if np.random.rand() < self.epsilon:
            return np.random.choice(actions)
        else:
            Q = self.qnet.forward(state)
            # extract legal actions
            legal_Q = [Q[0][a] for a in actions]
            idx = np.argmax(legal_Q)
            return actions[idx]

    def train(self, env: GameBoard, state: State, next_state: State = None, reward: int = None, done: bool = True):
        with tf.GradientTape() as tape:
            actions = env.get_actions(state, self.color)
            Q = self.qnet.forward(state)
            # extract legal actions
            legal_Q = [Q[0][a] for a in actions]
            Q = tf.reduce_max(legal_Q)

            if next_state is not None and reward is not None and not done:
                # proceed state by opponent
                next_actions = env.get_actions(next_state, -self.color)
                next_Q = self.qnet.forward(next_state)
                next_legal_Q = [next_Q[0][a] for a in next_actions]
                next_Q = tf.reduce_max(next_legal_Q)
                Q_target = reward + (1 - done) * self.gamma * next_Q
            else:
                Q_target = 0  # or an appropriate reward for game ending
            loss = tf.reduce_sum((Q - Q_target) **2)

        self.qnet.backward(tape, loss)
        return loss

    def step(self, observation, reward, done):
        pass
