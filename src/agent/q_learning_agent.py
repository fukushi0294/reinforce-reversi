from common.state import State
from agent.simple_agent import SimpleAgent
from env.gameboard import GameBoard
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.models import Sequential
from keras.optimizers import SGD
from common.exception import NotFoundLegalActionException
from typing import List
import pandas as pd


class QNet:
    def __init__(self) -> None:
        self.model = Sequential([
            Dense(512, activation='tanh', input_shape=(100, 64)),
            Dropout(0.5),
            Dense(512, activation='tanh'),
            Dropout(0.5),
            Dense(64, activation='tanh')
        ])
        self.optimizer = SGD(learning_rate=0.001)
        checkpoint = tf.train.Checkpoint(
            model=self.model, optimizer=self.optimizer)
        self.manager = tf.train.CheckpointManager(
            checkpoint, directory='./checkpoints/single', max_to_keep=3)

    def forward(self, state: State):
        X = state.board.flatten().reshape(1, -1)
        return self.model(X, training=True)

    def batch_forward(self, boards: np.ndarray):
        X = np.array([b.flatten().reshape(1, -1) for b in boards])
        if X.ndim == 1:
            return self.model(X[0])
        return self.model(X)

    def backward(self, tape: tf.GradientTape, loss):
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))

    def save(self):
        self.manager.save()


class MultiQNet:
    def __init__(self) -> None:
        self.models = [
            Sequential([
                Dense(36, activation='tanh', input_shape=(64,)),
                Dense(1, activation='tanh')
            ]) for _ in range(64)
        ]
        self.optimizer = SGD(learning_rate=0.001)
        checkpoint = tf.train.Checkpoint(
            model=self.models, optimizer=self.optimizer)
        self.manager = tf.train.CheckpointManager(
            checkpoint, directory='./checkpoints/multi', max_to_keep=3)

    def forward(self, env: GameBoard, state: State, actions: List[int]):
        action_to_state = {}
        for a in actions:
            next_state, _, _ = env.step(state, a, self.color)
            action_to_state[a] = next_state
        Qs = []
        for k, v in action_to_state.items():
            X = v.board.flatten().reshape(1, -1)
            Qs.append(self.models[k](X, training=True))
        return Qs

    def backward(self, tape: tf.GradientTape, loss, action):
        model = self.models[action]
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))


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

    def train(self, env: GameBoard, state: State, action: int, next_state: State, reward: int):
        with tf.GradientTape() as tape:
            Q = self.qnet.forward(state)
            next_actions = env.get_actions(next_state, self.color)
            q = Q[0, action]
            tape.watch(q)
            done = next_state.is_done()
            if not done and len(next_actions) != 0:
                next_Q = self.qnet.forward(next_state)
                next_legal_Q = [next_Q[0][a] for a in next_actions]
                q_target = reward + self.gamma * \
                    tf.reduce_max(next_legal_Q, axis=-1)
            else:
                q_target = reward
            loss = (q - q_target)**2

        self.qnet.backward(tape, loss)
        return loss.numpy()

    def batch_train(self, df: pd.DataFrame):
        current_board = df["current_board"].to_numpy()
        actions = df["action"].to_numpy()
        next_board = df["next_board"].to_numpy()

        with tf.GradientTape() as tape:
            Q = self.qnet.batch_forward(current_board)
            Q = tf.squeeze(Q, axis=1)
            q = tf.gather_nd(Q, list(enumerate(actions)))
            q = tf.cast(tf.expand_dims(q, axis=-1), dtype=tf.float64)
            tape.watch(q)

            next_Q = self.qnet.batch_forward(next_board).numpy()
            next_Q = np.squeeze(next_Q, axis=1)
            df_q = pd.DataFrame({'next_Q': [next_q for next_q in next_Q]})
            df_q["reward"] = df["reward"]
            df_q["next_actions"] = df["next_actions"]

            q_target = df_q.apply(self.get_max_Q, axis=1)
            q_target = tf.convert_to_tensor(q_target, dtype=tf.float64)
            loss = tf.reduce_mean(tf.square(tf.subtract(q, q_target)))

        self.qnet.backward(tape, loss)
        return loss.numpy()

    def get_max_Q(self, row):
        next_Q = row["next_Q"]
        reward = row["reward"]
        actions = row["next_actions"]
        if len(actions) > 0:
            Q = np.max([next_Q[int(a)] for a in actions])
            return reward + self.gamma * Q
        else:
            return reward

    def step(self, observation, reward, done):
        pass


class MultiQLearningAgent:
    def __init__(self, player_color: int = 1, agent: SimpleAgent = None):
        self.color = player_color
        self.gamma = 1
        self.alpha = 0.8
        self.qnet = MultiQNet()
        self.opponent = agent

    def get_action(self, env: GameBoard, state: State, game_cnt: int):
        actions = env.get_actions(state, self.color)
        if len(actions) == 0:
            raise NotFoundLegalActionException("No legal action")

        temperature = 0.9999995 ** game_cnt
        Qs = self.qnet.forward(env, state, actions)
        logits = tf.divide(Qs, temperature)
        action_probabilities = tf.nn.softmax(logits)
        return np.random.choice(actions, p=action_probabilities.numpy())

    def train(self, env: GameBoard, state: State, action: int, next_state: State, reward: int):
        with tf.GradientTape() as tape:
            Q = self.qnet.forward({action: state})
            next_actions = env.get_actions(next_state, self.color)
            q = Q[0]
            tape.watch(q)
            done = next_state.is_done()
            if not done and len(next_actions) != 0:
                next_Qs = self.qnet.forward(env, next_state, next_actions)
                q_target = reward + self.gamma * \
                    tf.reduce_max(next_Qs, axis=-1)
            else:
                q_target = reward
            loss = (q - q_target)**2

        self.qnet.backward(tape, loss, action)
        return loss.numpy()
