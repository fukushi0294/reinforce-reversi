
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from common.state import State


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

    def reward(self, state: State, action: int, next_state: State):
        return state.neighers_map

    def reset(self) -> State:
        board = np.zeros((8, 8), dtype=int)
        board[3][3], board[4][4] = 1, 1
        board[3][4], board[4][3] = -1, -1

        neighers = np.array([[2, 3], [2, 4], [3, 2], [3, 5], [
                            4, 2], [4, 5], [5, 3], [5, 4]])
        neighers_map = np.zeros((8, 8), dtype=bool)
        for n in neighers:
            row, col = n
            neighers_map[row, col] = True
        return State(board, neighers_map)

    # return next_state, reward, done
    def step(self, state: State, action: int, block):
        next_state = State(state.board, state.neighers_map)
        row, col = divmod(abs(action), 8)
        current = state.board[row][col]
        if current != 0:
            return next_state, 0, False

        # below process is legal action
        next_state.board[row, col] = block
        to_flipped = self.search(row, col, next_state.board, block)
        for r, c in to_flipped:
            next_state.board[r, c] *= -1
        next_state.board[row, col] = block
        self.update_neighers(next_state, row, col)
        return next_state, len(to_flipped), next_state.is_done()

    def search(self, start_row, start_col, board: np.ndarray, color: int):
        to_flipped = []
        for vec in self.action_vectors.values():
            row, col = start_row, start_col
            to_flipped_vec = []
            sandwitched = False
            while not sandwitched:
                row += vec[0]
                col += vec[1]
                if row < 0 or row > 7 or col < 0 or col > 7:
                    break
                searched = board[(row, col)]
                if searched == 0:
                    break
                if searched == color:
                    sandwitched = len(to_flipped_vec) > 0
                    break
                else:
                    to_flipped_vec.append((row, col))
            if sandwitched:
                to_flipped.extend(to_flipped_vec)
        return to_flipped

    def update_neighers(self, state: State, row, col):
        state.neighers_map[row, col] = False
        for vec in self.action_vectors.values():
            neigher_row = row + vec[0]
            neigher_col = col + vec[1]
            if neigher_row not in range(8) or neigher_col not in range(8):
                continue
            is_in = state.neighers_map[neigher_row, neigher_col]
            if state.board[neigher_row, neigher_col] == 0 and not is_in:
                state.neighers_map[neigher_row, neigher_col] = True

    def get_actions(self, state: State, color: int):
        to_flippeds = []
        rows, cols = np.where(state.neighers_map)
        for i in range(len(rows)):
            row, col = rows[i], cols[i]
            to_filpped = self.search(row, col, state.board, color)
            if len(to_filpped) > 0:
                to_flippeds.append((row, col))
        to_flippeds = set(to_flippeds)
        return list(map(lambda x: 8*x[0] + x[1], to_flippeds))

    def render(self, state: State):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots()

        cmap = mcolors.ListedColormap(['white', 'green', 'black'])
        bounds = [-1.5, -0.1, 0.1, 1.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        board_display = [[x for x in row] for row in state.board]

        self.ax.cla()
        self.ax.set_xticks(range(len(state.board)))
        self.ax.set_yticks(range(len(state.board[0])))
        self.ax.grid(True, which='both')
        self.ax.imshow(board_display, cmap=cmap, extent=[0, 8, 0, 8],
                       norm=norm, interpolation='none')
        plt.show()

    def render_v(self):
        pass


if __name__ == '__main__':
    env = GameBoard()
    state = env.reset()
    env.render(state)
