from copy import deepcopy
import numpy as np
from typing import Callable

from yinsh import GameState, YinshGame, connected, get_points_between, coords

BIG_NUMBER = (2**31) - 1
EPS = 1e-5


def norm2(coord):
    return coord[0] ** 2 + coord[1] ** 2


def actions(game_state):
    return game_state.valid_moves


def terminal_value(points):
    """return +1 if player 1 wins, -1 if player 2 wins, 0 if draw."""
    return np.sign(points[0] - points[1])


class FixedDepthMiniMaxTreePlayer:
    """Player that searches the game tree and picks the minimax optimal move according to heuristic value."""

    def __init__(
        self,
        player_number,
        depth,
        estimate_value: Callable[[GameState], float],
        opening_depth=1,
        ab=True,
    ):
        self.rng = np.random.default_rng()
        self.player = player_number
        self.estimate_value = estimate_value
        self.max_depth = depth
        self.opening_depth = opening_depth
        self.ab = ab

    def sort_moves(self, game_state):
        if game_state.turn_type != "remove run":
            return sorted(game_state.valid_moves, key=norm2)
        return game_state.valid_moves

    def negamax(self, game_state, depth, player_sign):
        if game_state.terminal or depth == 0:
            return player_sign * self.estimate_value(game_state)
        value = -float("inf")
        moves = self.sort_moves(game_state)
        for move in moves:
            g = YinshGame(deepcopy(game_state))
            g.take_turn(move)
            value = max(
                [value, -self.negamax(g.get_game_state(), depth - 1, -player_sign)]
            )
        return value

    def negamax_ab_prune(self, game_state, depth, a, b, player_sign):
        if game_state.terminal or depth == 0:
            return player_sign * self.estimate_value(game_state)
        value = -float("inf")
        moves = self.sort_moves(game_state)
        for move in moves:
            g = YinshGame(deepcopy(game_state))
            g.take_turn(move)
            sign_mult = [1, -1][g.active_player != game_state.active_player]
            value = max(
                [
                    value,
                    sign_mult
                    * self.negamax_ab_prune(
                        g.get_game_state(),
                        depth - 1,
                        sign_mult * b,
                        sign_mult * a,
                        sign_mult * player_sign,
                    ),
                ]
            )
            a = max([a, value])
            if a >= b:
                break
        return value

    def get_move_values(self, moves, game_state):
        if (
            game_state.turn_type == "setup new rings"
            and len(game_state.board.rings[game_state.active_player]) <= 3
        ):
            depth = self.opening_depth
        elif (
            game_state.turn_type == "setup new rings"
            and len(game_state.board.rings[game_state.active_player]) == 4
        ):
            depth = self.opening_depth + 1
        else:
            depth = self.max_depth
        values = np.zeros((len(moves),))
        for i, move in enumerate(moves):
            g = YinshGame(deepcopy(game_state))
            g.take_turn(move)
            player_sign = [1, -1][self.player]
            if self.ab:
                values[i] = self.negamax_ab_prune(
                    g.get_game_state(), depth, -float("inf"), float("inf"), player_sign
                )
            else:
                values[i] = self.negamax(g.get_game_state(), depth, player_sign)
        return values

    def make_move(self, game_state):
        moves = self.sort_moves(game_state)
        values = self.get_move_values(moves, game_state)
        # if len(values) != len(moves):
        #     print("debug size mismatch")
        #     print(f"debug values {values}")
        #     print(f"debug moves {moves}")
        # if values.size > 0:
        max_inds = []
        max_val = -float("inf")
        for i, value in enumerate(values):
            if value > max_val:
                max_inds = [i]
                max_val = value
            elif value == max_val:
                max_inds.append(i)
        # if len(max_inds) > 0:
        # print(f"DEBUG options: {[(values[i], moves[i]) for i in max_inds]}")
        # print(len(max_inds))
        # print(len(values))
        move = moves[self.rng.choice(max_inds)]

        # print(sorted(values))
        if type(game_state.valid_moves[0]) == tuple:
            return tuple(move)
        else:
            return [tuple(coord) for coord in move]
        # else:
        #     print("empty values")
        #     print(game_state)
        #     return


def num_ring_connections(game, rings) -> int:
    """Return number of coordinates on lines from rings.

    ring heuristic: number of coords in line with >= 1 player ring
    way grosser than haskell version,
    but more efficient than direct copy of its implementation into python
    since I don't think `any` would lazy evaluate while constructing the list.
    """
    connections = set()
    for c in coords:
        for r in rings:
            if connected(c, r):
                connections.add(c)
                break
    return len(connections)


def markers_between(game, start, end) -> int:
    """Return number of markers strictly between start and end."""
    return len([c for c in get_points_between(start, end) if game.is_marker(c)])


def controlled_markers(game, start_coord) -> int:
    """Return number of flipable markers via moving ring from start_coord."""
    return [
        markers_between(game, start_coord, end_coord)
        for end_coord in game.ring_moves(start_coord)
    ]


def num_controlled_markers(game, rings):
    # ring heuristic: sum of markers "controlled" by player's rings
    return sum([sum(controlled_markers(game, r)) for r in rings])


def num_unique_controlled_markers(game, rings):
    unique_controlled_markers = set()
    # ring heuristic: sum of markers "controlled" by player's rings
    cm_list = [controlled_markers(game, r) for r in rings]
    for cm in cm_list:
        for m in cm:
            unique_controlled_markers.add(m)
    return len(unique_controlled_markers)


def markers_x10(game, player):
    # marker heuristic: 10 * number of markers player has
    return 10 * len(game.get_markers(player))


def total_ring_moves(game, rings):
    # ring heuristic: sum up the number of ring moves each ring has for player
    num_ring_moves = sum([len(game.ring_moves(r)) for r in rings])
    return num_ring_moves


def combined_heuristic(heuristic_iterable, wts=None):
    if wts is None:

        def new_heuristic(game, pieces):
            return sum([heuristic(game, pieces) for heuristic in heuristic_iterable])

    else:

        def new_heuristic(game, pieces):
            return sum(
                [
                    wt * heuristic(game, pieces)
                    for wt, heuristic in zip(wts, heuristic_iterable)
                ]
            )

    return new_heuristic


def floyd_estimate(
    game_state,
    marker_heuristic=markers_x10,
    ring_heuristic=num_controlled_markers,
):
    """Python implementation of David Peter's Floyd AI's heuristic.

    https://github.com/sharkdp/yinsh/blob/master/src/Floyd.hs
    """
    if game_state.terminal:
        return terminal_value(game_state.points) * BIG_NUMBER

    yg = YinshGame(game_state)

    def current_and_immediate_points(player):
        # current points plus any points about to happen due to already made moves
        if (
            yg.turn_type == "remove run" or yg.turn_type == "remove ring"
        ) and yg.active_player == player:
            yg.points[player] + 1
        return yg.points[player]

    def value_markers(player):
        return marker_heuristic(yg, player)

    def value_rings(player):
        rings = yg.get_rings(player)
        return ring_heuristic(yg, rings)

    def value_points(player):
        return 100000 * current_and_immediate_points(player)

    def player_value(player):
        return value_points(player) + value_markers(player) + value_rings(player)

    return player_value(0) - player_value(1)
