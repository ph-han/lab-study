from collections import defaultdict
from config import UP, DOWN, LEFT, RIGHT

class Agent:
    gamma = 0.9

    def __init__(self, name: str):
        self.name = name
        self.V = defaultdict(lambda: 0.0)
        
        self.pi = defaultdict(lambda: {
            UP: 0.25,
            DOWN: 0.25,
            LEFT: 0.25,
            RIGHT: 0.25
        })