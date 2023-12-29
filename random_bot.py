import numpy as np


class UniformRandomPlayer:
    def __init__(self, rng=None):
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def make_move(self, game_state):
        move = self.rng.choice(game_state.valid_moves)

        if type(game_state.valid_moves[0]) == tuple:
            return tuple(move)
        else:
            return [tuple(coord) for coord in move]
