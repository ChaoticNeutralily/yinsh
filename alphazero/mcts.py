from copy import deepcopy
import logging

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


def Es_value(end, points):
    """Update Es[s] +1 if player 1 won, -1 if player 2, and 0 if not ended, eps if draw."""
    if not end:
        return 10  # arbitrary not finished
    elif points[0] > points[1]:
        return 1
    elif points[0] < points[1]:
        return -1
    else:
        return 0


class MCTS:
    """
    This class handles the Monte Carlo tree search tree.

    Directly adapted from https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times state s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game_state.terminal ended for state s
        self.Vs = {}  # stores game_state.valid_moves for state s
        self.rng = np.random.default_rng()

    def get_action_probability(self, game_state, history, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        game_state.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.num_MCTS_sims):
            self.search(game_state, history)

        s = str(game_state)
        counts = [self.Nsa.get((s, a), 0) for a in range(len(game_state.valid_moves))]

        if temp == 0:
            best_actions = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_action = self.rng.choice(best_actions)
            probs = [0] * len(counts)
            probs[best_action] = 1
            return probs

        # print(f"DEBUG counts:{counts}")
        counts = [x ** (1.0 / temp) for x in counts]
        # print(f"DEBUG counts temped:{counts}")
        counts_sum = float(sum(counts))

        probs = [x / counts_sum for x in counts]
        return probs

    def highest_prob_move(self, game_state, history):
        return game_state.valid_moves[
            np.argmax(self.get_action_probability(game_state, history, temp=0))
        ]

    def search(self, game_state, history, n=0):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = str(game_state)
        # print(f"DEBUG search {n} {s} \n{[h.rings for h in history]}")
        if self.Es.get(s, None) is None:
            # updates Es[s]. 10 if not ended, +1 if player 1 won, -1 if player 2 won, and 0 if draw
            self.Es[s] = Es_value(game_state.terminal, game_state.points)
        if self.Es[s] != 10:
            # terminal node
            return -self.Es[s]

        if self.Ps.get(s, None) is None:
            # leaf node
            policy, value = self.nnet.predict(game_state, history)
            self.Ps[s] = self.nnet.nnet.postprocess(policy, game_state.valid_moves)

            self.Vs[s] = game_state.valid_moves
            self.Ns[s] = 0
            # print(f"DEBUG value {value[0]}")
            # print(f"DEBUG value {type(value)}")
            return -value[0]

        valids = self.Vs[s]
        cur_best = -float("inf")

        # pick the action with the highest upper confidence bound
        for move_ind, move in enumerate(valids):
            if self.Qsa.get((s, move_ind)) is not None:
                upper_conf = self.Qsa[(s, move_ind)] + self.args.cpuct * self.Ps[s][
                    move_ind
                ] * np.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, move_ind)])
            else:
                upper_conf = (
                    self.args.cpuct * self.Ps[s][move_ind] * np.sqrt(self.Ns[s] + EPS)
                )  # Q = 0 ?

            if upper_conf > cur_best:
                cur_best = upper_conf
                best_move = move_ind

        # original specified the player that made the move,
        # but I'm using noncanonical gamestate with active player,
        # so I think it's not needed.
        # Only the strings are canonical gamestate returned by GameState's __repr__

        next_s = self.game.get_next_game_state(deepcopy(game_state), valids[best_move])
        if history:
            history = history[1:]
            history.append(deepcopy(next_s.board))

        # print(f"DEBUG next {(n+1)} search{str(next_s)}")
        value = 0.999 * self.search(next_s, deepcopy(history), n + 1)

        if self.Qsa.get((s, best_move), None) is not None:
            self.Qsa[(s, best_move)] = (
                self.Nsa[(s, best_move)] * self.Qsa[(s, best_move)] + value
            ) / (self.Nsa[(s, best_move)] + 1)
            self.Nsa[(s, best_move)] += 1
        else:
            self.Qsa[(s, best_move)] = value
            self.Nsa[(s, best_move)] = 1

        self.Ns[s] += 1
        return -value
