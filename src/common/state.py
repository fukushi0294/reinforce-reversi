from dataclasses import dataclass
import numpy as np
from collections import deque


@dataclass
class State:
    board: np.ndarray
    reward_map: np.ndarray
    color: int
    memory = deque(maxlen=5)

    @property
    def height(self):
        return len(self.reward_map)

    @property
    def width(self):
        return len(self.reward_map[0])

    @property
    def shape(self):
        return self.reward_map.shape

    def is_win(self, color) -> bool:
        return np.count_nonzero(self.board == color) > np.count_nonzero(self.board == - color)

    def is_done(self) -> bool:
        nonzero_elements = self.board[self.board != 0]
        if len(nonzero_elements) == 0:
            True
        if all(n == 1 for n in nonzero_elements) or all(n == -1 for n in nonzero_elements):
            True
        return False

    def legal_hands(self):
        if self.color > 0:
            hands = np.where(self.reward_map > 0)
        else:
            hands = np.where(self.reward_map < 0)
        return [(hands[0][i], hands[1][i]) for i in range(len(hands[0]))]

    def legal_actions(self):
        return [8*h[0] + h[1] for h in self.legal_hands()]

    def save(self, action: int):
        self.memory.append(action)
