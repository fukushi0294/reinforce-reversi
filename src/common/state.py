import numpy as np
from collections import deque
from copy import deepcopy


class State:
    def __init__(self, board: np.ndarray, neighers_map: np.ndarray, next_turn: int):
        self.board = board
        self.neighers_map = neighers_map
        self.next_turn = next_turn
        self.skip_cnt = 0
        self.memory = deque(maxlen=5)

    def skip(self):
        next_sate = deepcopy(self)
        next_sate.next_turn *= -1
        next_sate.skip_cnt += 1
        return next_sate

    @property
    def height(self):
        return len(self.neighers_map)

    @property
    def width(self):
        return len(self.neighers_map[0])

    @property
    def shape(self):
        return self.neighers_map.shape

    def is_win(self, color) -> bool:
        return np.count_nonzero(self.board == color) > np.count_nonzero(self.board == - color)

    def count(self, color) -> int:
        return np.count_nonzero(self.board == color)

    def is_done(self) -> bool:
        if self.skip_cnt == 2:
            return True
        return (self.board.size - np.count_nonzero(self.board)) == 0

    def save(self, action: int):
        self.memory.append(action)
