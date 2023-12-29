import numpy as np


class UniformRandomPlayer:
    def __init__(self, rng=None):
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def make_move(self, valid_moves):
        move = self.rng.choice(valid_moves)

        if type(valid_moves[0]) == tuple:
            return tuple(move)
        else:
            return [tuple(coord) for coord in move]
