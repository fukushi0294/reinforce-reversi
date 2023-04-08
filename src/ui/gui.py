if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import tkinter as tk
from tkinter import ttk
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

        # place my canvas to upper right corner
        self.window = tk.Tk()
        self.window.geometry("800x800")
        self.canvas = tk.Canvas(self.window, width=500, height=500)
        self.canvas.pack(side=tk.RIGHT), self.canvas.place(x=20, y=20)
        self.draw_board()

        # place text laval next to canvas
        self.player1_label = tk.Label(self.window, text="Player 1")
        self.player1_label.pack(
            side=tk.RIGHT), self.player1_label.place(x=600, y=20)

        # place player selection next to canvas and under label
        self.player1 = ttk.Combobox(
            self.window, values=["Human", "Random", "MonteCarlo", "Q-Learning"], state="readonly")
        self.player1.pack(side=tk.RIGHT), self.player1.place(x=600, y=50)
        self.player1.bind("<<ComboboxSelected>>", self.player1_selected)

        self.player2_label = tk.Label(self.window, text="Player 2")
        self.player2_label.pack(
            side=tk.RIGHT), self.player2_label.place(x=600, y=100)

        self.player2 = ttk.Combobox(
            self.window, values=["Random", "MonteCarlo", "Q-Learning"], state="readonly")
        self.player2.pack(side=tk.RIGHT), self.player2.place(x=600, y=130)
        self.player2.bind("<<ComboboxSelected>>", self.player2_selected)

        self.start_button = tk.Button(
            self.window, text="Start", command=self.start)
        self.start_button.pack(
            side=tk.RIGHT), self.start_button.place(x=600, y=200)
        self.start_button.bind(
            "<Button-1>", self.start_game)

    def start(self):
        self.window.mainloop()

    def start_game(self, event=None):
        if self.player1.get() and self.player2.get():
            self.window.event_generate("<<Start>>")

    def player1_selected(self, event=None):
        agent_map = {
            "Random": RandomAgent,
        }

        if self.player1.get() == "Human":
            self.player_color = 1

            def event_handler(event):
                self.place_stone(event)
                self.window.after(
                    1000, lambda: self.window.event_generate("<<Player1TurnOver>>"))
            self.canvas.bind("<Button-1>", lambda event: event_handler(event))
        else:
            agent = agent_map[self.player1.get()](player_color=1)

            def event_handler(event, agent):
                self.agent_action(event, agent)
                self.window.after(
                    1000, lambda: self.window.event_generate("<<Player1TurnOver>>"))

            self.window.bind("<<Player2TurnOver>>",
                             lambda event: event_handler(event, agent))
            self.window.bind(
                "<<Start>>", lambda event: event_handler(event, agent))

    def player2_selected(self, event=None):
        agent_map = {
            "Random": RandomAgent,
        }
        agent = agent_map[self.player2.get()](player_color=-1)

        def event_handler(event, agent):
            self.agent_action(event, agent)
            self.window.after(
                1000, lambda: self.window.event_generate("<<Player2TurnOver>>"))
        self.window.bind("<<Player1TurnOver>>",
                         lambda event: event_handler(event, agent))

    def agent_action(self, event, agent):
        action = agent.get_action(self.env, self.state)
        next_state, _, _ = self.env.step(self.state, action, agent.color)
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
        actions = env.get_actions(self.state)
        if action not in actions:
            return
        next_state, _, _ = self.env.step(self.state, action, self.player_color)
        self.state = next_state
        self.draw_board()


if __name__ == '__main__':
    env = GameBoard()
    gui = ReversiGUI(env=env)
    gui.start()
