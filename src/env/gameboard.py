
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class GameBoard:
    """
    below is action map
    ---------------------------
      0  1  2  3  4  5  6  7
      8  9 10 11 12 13 14 15
      16 17 18 19 20 21 22 23
      24 25 26 27 28 29 30 31
      32 33 34 35 36 37 38 39
      40 41 42 43 44 45 46 47
      48 49 50 51 52 53 54 55
      56 57 58 59 60 61 62 63
    ---------------------------
    """

    def __init__(self):
        self.board = np.zeros((8, 8), dtype=int)
        self.board[3][3], self.board[4][4] = 1, 1
        self.board[3][4], self.board[4][3] = -1, -1
        self.reward_map = np.zeros((8, 8), dtype=int)
        self.reward_map[2, 3], self.reward_map[2, 4] = -1, 1
        self.reward_map[3, 2], self.reward_map[3, 5] = -1, 1
        self.reward_map[4, 2], self.reward_map[4, 5] = 1, -1
        self.reward_map[5, 3], self.reward_map[5, 4] = 1, -1

        self.action_vectors = {
            0: (-1, -1),  # UP_LEFT,
            1: (-1, 0),  # UP
            2: (-1, 1),  # UP_RIGHT
            3: (0, -1),  # LEFT
            4: (0, 1),  # RIGHT,
            5: (1, -1),  # DOWN_LEFT
            6: (1, 0),  # DOWN
            7: (1, 1)  # DOWN_RIGHT
        }

        self.turn = 1

        self.fig, self.ax = None, None

    @property
    def height(self):
        return len(self.reward_map)

    @property
    def width(self):
        return len(self.reward_map[0])

    @property
    def shape(self):
        return self.reward_map.shape

    def actions(self, color):
        if color > 0:
            actions = np.where(self.reward_map > 0)
        else:
            actions = np.where(self.reward_map < 0)
        return [(actions[0][i], actions[1][i]) for i in range(len(actions[0]))]

    def states(self):
        return self.turn, self.actions(self.turn)

    def reward(self, state, action, next_state):
        return self.reward_map[next_state]

    def reset(self):
        self.board = np.zeros((8, 8), dtype=int)
        self.board[3][3], self.board[4][4] = 1, 1
        self.board[3][4], self.board[4][3] = -1, -1
        self.reward_map = np.zeros((8, 8), dtype=int)
        self.reward_map[2, 3], self.reward_map[2, 4] = -1, 1
        self.reward_map[3, 2], self.reward_map[3, 5] = -1, 1
        self.reward_map[4, 2], self.reward_map[4, 5] = 1, -1
        self.reward_map[5, 3], self.reward_map[5, 4] = 1, -1

    # return next_state, reward, done
    def step(self, action: int):
        # TODO: state is not defined currentry
        next_state = None
        block = 1 if action > 0 else -1
        row, col = divmod(abs(action), 8)
        current = self.board[row][col]
        if current != 0:
            return next_state, 0, False

        # check illegal step
        reward = self.reward_map[row, col]
        if reward == 0 or reward * block < 0:
            return next_state, 0, False

        # below process is legal action
        to_flipped = self.search(row, col, block)
        for row, col in to_flipped:
            self.board[row, col] *= -1
        self.update_reward(row, col, block)
        self.board[row, col] = block
        self.turn *= -1
        return next_state, reward, self.is_done()

    def is_done(self) -> bool:
        nonzero_elements = self.board[self.board != 0]
        if len(nonzero_elements) == 0:
            True
        if all(n == 1 for n in nonzero_elements) or all(n == -1 for n in nonzero_elements):
            True
        return False

    def search(self, start_row, start_col, color):
        to_flipped = []
        for vec in self.action_vectors.values():
            row, col = start_row, start_col
            while True:
                row += vec[0]
                col += vec[1]
                if row < 0 or row > 7 or col < 0 or col > 7:
                    break
                searched = self.board[(row, col)]
                if searched == 0 or searched == color:
                    break
                to_flipped.append((row, col))
        return to_flipped

    def update_reward(self, row, col, color):
        self.reward_map[row, col] = 0
        for vec in self.action_vectors.values():
            neigher_row = row + vec[0]
            neigher_col = col + vec[1]
            to_filpped = self.search(neigher_row, neigher_col, color)
            self.reward_map[neigher_row, neigher_col] = len(to_filpped)

    def render(self):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots()

        cmap = mcolors.ListedColormap(['white', 'green', 'black'])
        bounds = [-1.5, -0.1, 0.1, 1.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        board_display = [[x for x in row] for row in self.board]

        self.ax.cla()
        self.ax.set_xticks(range(len(self.board)))
        self.ax.set_yticks(range(len(self.board[0])))
        self.ax.grid(True, which='both')
        self.ax.imshow(board_display, cmap=cmap, extent=[0, 8, 0, 8],
                       norm=norm, interpolation='none')
        plt.show()

    def render_v(self):
        pass


if __name__ == '__main__':
    env = GameBoard()
    env.render()
