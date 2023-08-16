import argparse
import os
import numpy as np
from collections import namedtuple
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from network.cnn5 import DQN
from env.gameboard import GameBoard

training_data = np.fromfile(args.training_data, TrainingData)
memory = ReplayMemory(training_data)

BATCH_SIZE = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_net = DQN().to(device)

optimizer = optim.RMSprop(target_net.parameters(), lr=1e-5)
#optimizer = optim.SGD(target_net.parameters(), lr=0.01, momentum=0.9)


######################################################################
# Training loop

board = GameBoard()
features = np.empty((BATCH_SIZE, 2, 8, 8), np.float32)

loss_sum = 0
log_interval = 100
for batch_idx in range(10000):
    transitions = memory.sample(BATCH_SIZE)
    moves = []
    for i, data in enumerate(transitions):
        board.set_bitboard(data['bitboard'], data['turn'])
        move = data['move']
        assert board.is_legal(move)
        n = random.randint(0, 1)
        if n == 0:
            board.piece_planes(features[i])
            moves.append(move)
        elif n == 1:
            board.piece_planes_rotate180(features[i])
            moves.append(move_rotate180(move))

    # https://github.com/TadaoYamaoka/creversi_gym/blob/master/creversi_gym/ggf_to_training_data.py#L54-L66
    # 同一のゲームのmovesのイテレーションの最中では結果は変わらない
    # ゲームが途中であっても最終的な結果のみでrewardを決定している
    state_batch = torch.from_numpy(features).to(device)
    action_batch = torch.tensor(moves, device=device, dtype=torch.long).view(-1, 1)
    reward_batch = torch.tensor(transitions['reward'], device=device, dtype=torch.float32)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = target_net(state_batch).gather(1, action_batch)

    expected_state_action_values = reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in target_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    loss_sum += loss.item()
    if (batch_idx + 1) % log_interval == 0:
        print(f"loss = {loss_sum / log_interval}")
        loss_sum = 0

print('Complete')