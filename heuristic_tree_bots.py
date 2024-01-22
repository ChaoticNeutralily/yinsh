from copy import deepcopy
from functools import cache
from lru import LRU
import numpy as np
from typing import Callable

from yinsh import GameState, YinshGame, connected, get_points_between, coords
from alphazero.nn_utils import get_state_symmetries

BIG_NUMBER = (2**31) - 1
EPS = 1e-5


def norm2(coord):
    return coord[0] ** 2 + coord[1] ** 2


def actions(game_state):
    return game_state.valid_moves


def terminal_value(points):
    """return +1 if player 1 wins, -1 if player 2 wins, 0 if draw."""
    return np.sign(points[0] - points[1])


def proper_move_type(move, reference_move):
    if type(reference_move) == tuple:
        return tuple(move)
    return [tuple(coord) for coord in move]


class FixedDepthMiniMaxTreePlayer:
    """Player that searches the game tree and picks the minimax optimal move according to heuristic value."""

    def __init__(
        self,
        player_number,
        depth,
        estimate_value: Callable[[GameState], float],
        opening_depth=1,
        pruning="ab",  # also can be "scout",
        opening_estimate_value=None,
        opening_threshold=3,
    ):
        self.rng = np.random.default_rng()
        self.player = player_number
        self.estimate_value = estimate_value
        self.max_depth = depth
        self.opening_depth = opening_depth
        self.pruning = pruning
        if opening_estimate_value is None:
            self.opening_value = estimate_value
        else:
            self.opening_value = opening_estimate_value
        self.transposition_table = LRU(2**20)
        self.opening_threshold = opening_threshold

    def sort_moves(self, game_state):
        if game_state.turn_type != "remove run":
            return sorted(
                game_state.valid_moves, key=lambda m: len(num_rconnections(m))
            )
        return game_state.valid_moves

    def sorted_children(self, game_state, depth):
        children = [
            YinshGame.get_next_game_state(deepcopy(game_state), move)
            for move in game_state.valid_moves
        ]
        return sorted(
            children,
            key=lambda child_state: self.transposition_table.get(
                (
                    str(child_state),
                    child_state.turn,
                    depth - 1,
                ),
                0,
            ),
            reverse=True,
        )

    def negamax(self, game_state, depth, player_sign, estimate):
        if game_state.terminal or depth == 0:
            return player_sign * estimate(game_state)
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
                    * self.negamax(
                        g.get_game_state(), depth - 1, sign_mult * player_sign, estimate
                    ),
                ]
            )
        return value

    def negamax_ab_prune(self, game_state, depth, a, b, player_sign, estimate):
        a0 = a

        # use cache for game_state
        entry = self.transposition_table.get(
            (str(game_state), game_state.active_player), False
        )
        if entry and entry["depth"] >= depth:
            if entry["flag"] == "Exact":
                return entry["value"]
            elif entry["flag"] == "Lower":
                a = max(a, entry["value"])
            elif entry["flag"] == "Upper":
                b = min(b, entry["value"])
            if a >= b:
                return entry["value"]

        if game_state.terminal or depth == 0:
            return player_sign * estimate(game_state)

        children = self.sorted_children(game_state, depth)
        value = -float("inf")
        for child in children:
            sign_mult = [1, -1][child.active_player != game_state.active_player]
            state_value = self.negamax_ab_prune(
                child,
                depth - 1,
                sign_mult * b,
                sign_mult * a,
                sign_mult * player_sign,
                estimate,
            )
            value = max([value, sign_mult * state_value])
            a = max([a, value])
            if a >= b:
                break

        # store new node in cache
        if value <= a0:
            flag = "Upper"
        elif value >= b:
            flag = "Lower"
        else:
            flag = "Exact"
        self.transposition_table[(str(game_state), game_state.active_player)] = {
            "value": value,
            "depth": depth,
            "flag": flag,
        }
        return value

    def negascout(self, game_state, depth, a, b, player_sign, estimate):
        a0 = a

        # use cache for game_state
        entry = self.transposition_table.get(
            (str(game_state), game_state.active_player), False
        )
        if entry and entry["depth"] >= depth:
            if entry["flag"] == "Exact":
                return entry["value"]
            elif entry["flag"] == "Lower":
                a = max(a, entry["value"])
            elif entry["flag"] == "Upper":
                b = min(b, entry["value"])
            if a >= b:
                return entry["value"]

        if game_state.terminal or depth == 0:
            return player_sign * estimate(game_state)

        children = self.sorted_children(game_state, depth)
        value = -float("inf")
        for i, child in enumerate(children):
            sign_mult = [1, -1][child.active_player != game_state.active_player]
            if i == 0:
                state_value = sign_mult * self.negascout(
                    child,
                    depth - 1,
                    sign_mult * b,
                    sign_mult * a,
                    sign_mult * player_sign,
                    estimate,
                )
            else:
                state_value = sign_mult * self.negascout(
                    child,
                    depth - 1,
                    sign_mult * a - 1,
                    sign_mult * a,
                    sign_mult * player_sign,
                    estimate,
                )
                if a < state_value and state_value < b:
                    state_value = sign_mult * self.negascout(
                        child,
                        depth - 1,
                        sign_mult * b,
                        sign_mult * a,
                        sign_mult * player_sign,
                        estimate,
                    )
            value = max([value, state_value])
            a = max([a, value])
            if a >= b:
                break

        # store new node in cache
        if value <= a0:
            flag = "Upper"
        elif value >= b:
            flag = "Lower"
        else:
            flag = "Exact"
        self.transposition_table[(str(game_state), game_state.active_player)] = {
            "value": value,
            "depth": depth,
            "flag": flag,
        }
        return value

    def get_move_values(self, moves, game_state):
        if (
            game_state.turn_type == "setup new rings"
            and len(game_state.board.rings[game_state.active_player])
            <= self.opening_threshold
        ):
            depth = self.opening_depth
            estimate = self.opening_value
        elif (
            game_state.turn_type == "setup new rings"
            and len(game_state.board.rings[game_state.active_player])
            == self.opening_threshold + 1
        ):
            depth = self.opening_depth + 1
            estimate = self.opening_value
        else:
            depth = self.max_depth
            estimate = self.estimate_value
        values = np.zeros((len(moves),))
        for i, move in enumerate(moves):
            g = YinshGame(deepcopy(game_state))
            g.take_turn(move)
            player_sign = [1, -1][self.player]
            if self.pruning == "ab":
                values[i] = self.negamax_ab_prune(
                    g.get_game_state(),
                    depth,
                    -float("inf"),
                    float("inf"),
                    player_sign,
                    estimate,
                )
            elif self.pruning == "scout":
                values[i] = self.negascout(
                    g.get_game_state(),
                    depth,
                    -float("inf"),
                    float("inf"),
                    player_sign,
                    estimate,
                )
            else:
                values[i] = self.negamax(
                    g.get_game_state(), depth, player_sign, estimate
                )
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
        return proper_move_type(move, game_state.valid_moves[0])
        # else:
        #     print("empty values")
        #     print(game_state)
        #     return


@cache
def num_rconnections(ring) -> set:
    """Return number of coordinates on lines from rings.

    ring heuristic: number of coords in line with >= 1 player ring
    way grosser than haskell version,
    but more efficient than direct copy of its implementation into python
    since I don't think `any` would lazy evaluate while constructing the list.
    """
    connections = set()
    for c in coords:
        if connected(c, ring):
            connections.add(c)
    return connections


def num_ring_connections(game, rings):
    cons = set()
    for r in rings:
        cons = cons.union(num_rconnections(tuple(r)))
    return len(cons)


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

    return (player_value(0) - player_value(1)) * (0.999**game_state.turn)
