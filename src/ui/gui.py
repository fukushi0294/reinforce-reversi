if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import tkinter as tk
import numpy as np
from env.gameboard import GameBoard
from agent.random_agent import RandomAgent
from agent.simple_agent import SimpleAgent


class ReversiGUI:
    def __init__(self, env: GameBoard):
        self.env = env
        self.state = env.reset()
        self.board_size = self.state.board.shape[0]
        self.color_map = {
            1: 'black',
            -1: 'white'
        }

        self.window = tk.Tk()
        self.canvas = tk.Canvas(self.window, width=500, height=500)
        self.canvas.pack()
        self.draw_board()

    def start(self):
        self.window.mainloop()

    def bind_player(self, args: dict):
        for k, v in args.items():
            if isinstance(v, str) and "Human" == v:
                self.player_color = k
                self.canvas.bind("<Button-1>", self.place_stone)
                self.window.bind("<<HumanDone>>", self.agent_action)
            else:
                self.agent = v

    def agent_action(self, event):
        action = self.agent.get_action(self.env, self.state)
        next_state, _, _ = self.env.step(self.state, action, self.agent.color)
        self.state = next_state
        self.draw_board()

    def draw_board(self):
        square_size = 500 // self.board_size
        for i in range(self.board_size):
            for j in range(self.board_size):
                x1 = i * square_size
                y1 = j * square_size
                x2 = (i+1) * square_size
                y2 = (j+1) * square_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill='green')

        board = self.state.board
        space = square_size * 0.1
        for k, v in self.color_map.items():
            place = np.argwhere(board == k)
            for p in place:
                row, col = p
                self.canvas.create_oval(
                    col*square_size + space, row*square_size + space, (col+1)*square_size - space, (row+1)*square_size-space, fill=v)

    def place_stone(self, event):
        square_size = 500 // self.board_size
        # Get the row and column where the user clicked
        col = int(event.x / square_size)
        row = int(event.y / square_size)

        action = 8 * row + col
        next_state, _, _ = self.env.step(self.state, action, self.player_color)
        self.state = next_state
        self.draw_board()
        self.window.event_generate("<<HumanDone>>", when="tail")


if __name__ == '__main__':
    env = GameBoard()
    gui = ReversiGUI(env=env)

    agent = RandomAgent(player_color=-1)
    gui.bind_player({
        1: "Human",
        -1: agent
    })
    gui.start()
