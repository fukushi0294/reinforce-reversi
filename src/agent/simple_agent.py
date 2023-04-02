from abc import ABC, abstractmethod

class SimpleAgent(ABC):
    def __init__(self, player_color:int):
        self.color = player_color
    
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, observation, reward, done):
        pass
