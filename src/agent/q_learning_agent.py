from common.state import State
from agent.simple_agent import SimpleAgent
from env.gameboard import GameBoard
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from keras.optimizers import SGD
from common.exception import NotFoundLegalActionException


class QNet:
    def __init__(self) -> None:
        self.model = Sequential([
            Dense(44, activation='tanh', input_shape=(64,)),
            Dense(64, activation='tanh')
        ])
        self.optimizer = SGD(learning_rate=0.001)
        checkpoint = tf.train.Checkpoint(
            model=self.model, optimizer=self.optimizer)
        self.manager = tf.train.CheckpointManager(
            checkpoint, directory='./checkpoints', max_to_keep=3)

    def forward(self, state: State):
        X = state.board.flatten().reshape(1, -1)
        return self.model(X, training=True)

    def backward(self, tape: tf.GradientTape, loss):
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))

    def save(self):
        self.manager.save()


class QLearningAgent(SimpleAgent):
    def __init__(self, player_color: int = 1, agent: SimpleAgent = None):
        self.color = player_color
        self.gamma = 1
        self.alpha = 0.8
        self.epsilon = 0.1
        self.qnet = QNet()
        self.opponent = agent

    def reset(self):
        pass

    def get_action(self, env: GameBoard, state: State):
        actions = env.get_actions(state, self.color)
        if len(actions) == 0:
            raise NotFoundLegalActionException("No legal action")

        if np.random.rand() < self.epsilon:
            return np.random.choice(actions)
        else:
            Q = self.qnet.forward(state)
            # extract legal actions
            legal_Q = [Q[0][a] for a in actions]
            idx = np.argmax(legal_Q)
            return actions[idx]
        
    def get_action_by_boltzmann(self, env: GameBoard, state: State, game_cnt: int):
        actions = env.get_actions(state, self.color)
        if len(actions) == 0:
            raise NotFoundLegalActionException("No legal action")
        temperature = 0.995 ** game_cnt
        Q = self.qnet.forward(state)
        legal_Q = [Q[0][a] for a in actions]
        logits = tf.divide(legal_Q, temperature)
        action_probabilities = tf.nn.softmax(logits)
        return np.random.choice(actions, p=action_probabilities.numpy())


    def train(self, env: GameBoard, state: State, action:int, next_state: State, reward: int):
        with tf.GradientTape() as tape:
            Q = self.qnet.forward(state)
            next_actions = env.get_actions(next_state, self.color)
            q = Q[0, action]
            done = next_state.is_done()
            if not done and len(next_actions) != 0:
                next_Q = self.qnet.forward(next_state)
                next_legal_Q = [next_Q[0][a] for a in next_actions]
                q_target = reward + self.gamma * tf.reduce_max(next_legal_Q, axis=-1)
            else:
                q_target = reward
            loss = (q - q_target)**2

        self.qnet.backward(tape, loss)
        return loss.numpy()

    def step(self, observation, reward, done):
        pass
