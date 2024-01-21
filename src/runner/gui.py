if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import tkinter as tk
from tkinter import ttk
from agent.monte_carlo_agent import MonteCarloAgent
import numpy as np
from env.gameboard import GameBoard
from agent.random_agent import RandomAgent
from common.exception import NotFoundLegalActionException


class ReversiGUI:
    def __init__(self, env: GameBoard, agent_delay=1000):
        self.env = env
        self.state = env.reset()
        self.delay = agent_delay
        self.board_size = self.state.board.shape[0]
        self.color_map = {
            1: 'black',
            -1: 'white'
        }

        # place my canvas to upper right corner
        self.window = tk.Tk()
        self.window.title("Reversi Simulator")
        self.window.geometry("800x800")
        self.canvas = tk.Canvas(self.window, width=500, height=500)
        self.canvas.pack(side=tk.RIGHT), self.canvas.place(x=20, y=20)
        self.draw_board()

        self.player1_agent = None
        # place text laval next to canvas
        self.player1_label = tk.Label(self.window, text="Player 1")
        self.player1_label.pack(
            side=tk.RIGHT), self.player1_label.place(x=600, y=20)

        # place player selection next to canvas and under label
        self.player1 = ttk.Combobox(
            self.window, values=["Human", "Random", "MonteCarlo", "Q-Learning"], state="readonly")
        self.player1.pack(side=tk.RIGHT), self.player1.place(x=600, y=50)
        self.player1.bind("<<ComboboxSelected>>", self.player1_selected)

        self.player2_agent = None
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

        # add reset button below start button
        self.reset_button = tk.Button(self.window, text="Reset")
        self.reset_button.pack(
            side=tk.RIGHT), self.reset_button.place(x=600, y=250)
        self.reset_button.bind(
            "<Button-1>", self.reset_game)

        # subscribe to done event and when recived, show message box
        self.window.bind("<<Done>>", self.done)

        self.listbox = tk.Listbox(self.window)
        self.listbox.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.listbox.place(
            x=10, y=540, width=0.9*self.window.winfo_screenwidth(), height=self.window.winfo_screenheight()-self.canvas.winfo_height() - 20)

    def start(self):
        self.window.mainloop()

    def log(self, message):
        self.listbox.insert(tk.END, message)
        self.listbox.see(tk.END)

    def done(self, event=None):
        black_count = self.state.count(1)
        white_count = self.state.count(-1)
        self.log(f"Black: {black_count}, White: {white_count}")
        if self.state.is_win(1):
            self.log("Black wins")
        elif self.state.is_win(-1):
            self.log("White wins")
        else:
            self.log("Draw")

    def start_game(self, event=None):
        if self.player1.get() and self.player2.get():
            self.window.event_generate("<<Start>>")
            self.log("Game started")

    def reset_game(self, event=None):
        self.state = self.env.reset()
        if self.player1_agent is not None:
            self.player1_agent.reset()
        if self.player2_agent is not None:
            self.player2_agent.reset()

        self.draw_board()
        self.log("Game reset")

    def player1_selected(self, event=None):
        agent_map = {
            "Random": RandomAgent,
            "MonteCarlo": MonteCarloAgent,
        }

        if self.player1.get() == "Human":
            action = lambda e: self.human_action(e, 1) 
            self.canvas.bind(
                "<Button-1>", lambda event: self.event_handler(event, action, 1))
        else:
            agent = agent_map[self.player1.get()](player_color=1)
            action = lambda e: self.agent_action(e, agent)
            self.player1_agent = agent
            self.window.bind("<<Player2TurnOver>>",
                             lambda event: self.event_handler(event, action, 1))
            self.window.bind(
                "<<Start>>", lambda event: self.event_handler(event, action, 1))

    def player2_selected(self, event=None):
        agent_map = {
            "Random": RandomAgent,
            "MonteCarlo": MonteCarloAgent,
        }
        agent = agent_map[self.player2.get()](player_color=-1)
        self.player2_agent = agent
        action = lambda e: self.agent_action(e, agent)
        self.window.bind("<<Player1TurnOver>>",
                         lambda event: self.event_handler(event, action, 2))

    def event_handler(self, event, player_action, player: int):
        assert player in [1, 2], "player argument should be 1 or 2"
        done = player_action(event)
        if done:
            self.window.event_generate("<<Done>>")
        else:
            self.window.after(
                self.delay, lambda: self.window.event_generate(f"<<Player{player}TurnOver>>"))

    def agent_action(self, _, agent):
        try:
            action = agent.get_action(self.env, self.state)
            next_state, _, done = self.env.step(
                self.state, action, agent.color)
            self.state = next_state
            self.draw_board()
            return done
        except NotFoundLegalActionException as e:
            self.log("No legal action, turn is skipped")
            opponent_actions = self.env.get_actions(self.state, -agent.color)
            if len(opponent_actions) != 0:
                self.state.next_turn *= -1
                return False
            else:
                return True

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

    def human_action(self, event, player):
        square_size = 500 // self.board_size
        # Get the row and column where the user clicked
        col = int(event.x / square_size)
        row = int(event.y / square_size)

        color = 1 if player == 1 else -1
        action = 8 * row + col
        actions = env.get_actions(self.state, color)
        if action not in actions:
            return
        next_state, _, done = self.env.step(
            self.state, action, color)
        self.state = next_state
        self.draw_board()
        return done


if __name__ == '__main__':
    env = GameBoard()
    gui = ReversiGUI(env=env, agent_delay=200)
    gui.start()
