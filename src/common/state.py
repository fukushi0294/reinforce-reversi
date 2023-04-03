from dataclasses import dataclass
import numpy as np


@dataclass
class State:
    board: np.ndarray
    reward_map: np.ndarray
    color: int

    @property
    def height(self):
        return len(self.reward_map)

    @property
    def width(self):
        return len(self.reward_map[0])

    @property
    def shape(self):
        return self.reward_map.shape

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
