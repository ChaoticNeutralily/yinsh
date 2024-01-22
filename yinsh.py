"""Implement the rules of Yinsh in Python. 

Heavily based on Haskell implemention by David Peter.
David's implementation at https://github.com/sharkdp/yinsh/tree/master
Rules of Yinsh at https://www.boardspace.net/yinsh/english/rules.htm
"""
from copy import deepcopy
from dataclasses import dataclass, field
from math import sqrt
from typing import List, Optional, Tuple, Union


def default_field(obj):
    return field(default_factory=lambda: deepcopy(obj))


MAX_MARKERS = 51

N = (0, 1)
NE = (1, 1)
SE = (1, 0)
S = (0, -1)
SW = (-1, -1)
NW = (-1, 0)

directions = [N, NE, SE, S, SW, NW]


## coordinate stuff
def valid_coord(x, y):
    return (0.5 * sqrt(3) * x) ** 2 + (0.5 * x - y) ** 2 <= 4.6**2


coords = [(x, y) for x in range(-5, 6) for y in range(-5, 6) if valid_coord(x, y)]


def adjacent(xy, xy2):
    return (xy[0] - xy2[0]) ** 2 + (xy[1] - xy2[1]) ** 2 == 1 or (
        (xy[0] - xy2[0]) == 1 and (xy[1] - xy2[1]) == 1
    )


def connected(xy, xy2):
    return (
        (xy[0] == xy2[0]) or (xy[1] == xy2[1]) or ((xy[0] - xy2[0]) == (xy[1] - xy2[1]))
    )


def add(xy, xy2):
    return (xy[0] + xy2[0], xy[1] + xy2[1])


def sub(xy, xy2):
    return (xy[0] - xy2[0], xy[1] - xy2[1])


def mul(xy, c):
    return (xy[0] * c, xy[1] * c)


def get_points_between(start, end):
    delta = sub(end, start)
    num = max(list(delta) + [-d for d in delta])
    step = (delta[0] // num, delta[1] // num)
    points = []
    for i in range(1, num):
        points.append(add(start, mul(step, i)))
    return points


def ray(xy, direction):
    line_points = []
    valid = True
    curr_point = xy
    while True:
        if valid_coord(*curr_point):
            line_points.append(curr_point)
            curr_point = add(curr_point, direction)
        else:
            return line_points


# N = (0, 1)
# NE = (1, 1)
# SE = (1, 0)
N_run_start_coords = [c for c in coords if valid_coord(*add(c, mul(N, 4)))]
NE_run_start_coords = [c for c in coords if valid_coord(*add(c, mul(NE, 4)))]
SE_run_start_coords = [c for c in coords if valid_coord(*add(c, mul(SE, 4)))]
run_start_coords = [
    set(N_run_start_coords),
    set(NE_run_start_coords),
    set(SE_run_start_coords),
]


class GameBoard:
    def __init__(self, elements={}, rings=[[], []], markers=[[], []]):
        self.elements = elements  # coord: ("ring"/"marker", player)
        self.rings = [[], []]
        self.markers = [[], []]

    def add_element(self, coordinate, element):
        self.elements[coordinate] = element
        if element[0] == "ring":
            self.rings[element[1]].append(coordinate)
        else:
            self.markers[element[1]].append(coordinate)

    def remove_element(self, coordinate):
        element = self.elements.pop(coordinate)
        if element is None:
            return
        if element[0] == "ring":
            self.rings[element[1]].remove(coordinate)
        else:
            self.markers[element[1]].remove(coordinate)

    def modify_element(self, coordinate, new_element):
        self.remove_element(coordinate)
        self.add_element(coordinate, new_element)

    def __repr__(self):
        return f"GameBoard(elements = {repr(self.elements)}, rings = {repr(self.rings)}, markers = {repr(self.markers)})"


@dataclass
class GameState:
    turn_type: str = "setup new rings"
    active_player: int = 0
    board: GameBoard = field(
        default_factory=lambda: GameBoard(elements={}, rings=[[], []], markers=[[], []])
    )
    points: List[int] = field(default_factory=lambda: [0, 0])
    points_to_win: int = 3
    valid_moves: Union[
        List[Tuple[int, int]], List[List[Tuple[int, int]]]
    ] = default_field(coords)
    last_moved: int = 0
    prev_ring: Tuple[int, int] = (0, 0)
    max_markers_before_draw: int = MAX_MARKERS
    terminal: bool = False
    turn: int = 0

    def prev_ring_str(self):
        if self.turn_type == "move_ring":
            return str(self.prev_ring)
        return ""

    def __repr__(self):
        """String of a canonical rep of the game."""
        return (
            str(int(self.active_player == self.last_moved))
            + str(int(self.terminal))
            + self.turn_type
            + str(
                [
                    self.board.markers[self.active_player],
                    self.board.markers[1 - self.active_player],
                    self.max_markers_before_draw,
                ]
            )
            + str(
                [
                    self.board.rings[self.active_player],
                    self.board.rings[1 - self.active_player],
                    self.prev_ring_str(),
                ]
            )
            + str(
                [
                    self.points[self.active_player],
                    self.points[1 - self.active_player],
                    self.points_to_win,
                ]
            )
        )


class YinshGame:
    def __init__(self, game_state: Optional[GameState] = None):
        if game_state is None:
            game_state = GameState()
        self.turn_type: str = game_state.turn_type
        self.active_player: int = game_state.active_player
        self.board: GameBoard = deepcopy(game_state.board)
        self.points: List[int] = deepcopy(game_state.points)
        self.points_to_win: int = game_state.points_to_win
        self.valid_moves: List = deepcopy(game_state.valid_moves)
        self.last_moved: int = game_state.last_moved
        self.prev_ring: Tuple[int, int] = game_state.prev_ring
        self.max_markers_before_draw: int = game_state.max_markers_before_draw
        self.terminal: bool = game_state.terminal
        self.turn: int = game_state.turn

    def get_game_state(self):
        return deepcopy(
            GameState(
                self.turn_type,
                self.active_player,
                self.board,
                self.points,
                self.points_to_win,
                sorted(self.valid_moves),
                self.last_moved,
                self.prev_ring,
                self.max_markers_before_draw,
                self.is_terminal(),
                self.turn,
            )
        )

    def get_markers(self, player):
        return self.board.markers[player]

    def get_rings(self, player):
        return self.board.rings[player]

    def get_element(self, coordinate):
        return self.board.elements.get(coordinate)  # None if nothing there

    def is_marker(self, coordinate):
        e = self.get_element(coordinate)
        return e is not None and e[0] == "marker"

    def is_ring(self, coordinate):
        e = self.get_element(coordinate)
        return e is not None and e[0] == "ring"

    def is_free(self, coordinate):
        return self.get_element(coordinate) is None

    def ring_moves_in_direction(self, coordinate, direction):
        """Get allowed coords for a ring to move along a given direction
        This is so gross.
        Rules:
        - A ring must always move in a straight line.
        - A ring must always move to a vacant space.
        - A ring may move over one or more vacant spaces.
        - A ring may jump over one or more markers, regardless of color,
        as long as they are lined up without interruption.
        In other words, if you jump over one or more markers,
        you must always put your ring in the first vacant space directly behind
        the markers you jumped over.
        - A ring may first move over one or more vacant spaces and continue with
        a jump over one or more markers.
        But, as stated above, after jumping over one or more markers,
        it may not move over any more vacant spaces.
        - A ring can only jump over markers, not over rings.
        """
        line = ray(coordinate, direction)[1:]
        free = []
        rest = []
        add_to_free = True
        for p in line:
            if add_to_free and self.is_free(p):
                free.append(p)
            else:
                # there is an obstacle.
                add_to_free = False
                # at least one marker if we're here. rest = next free spot
                if self.is_free(p):
                    # if the space is free, we've hopped over a marker
                    rest.append(p)
                    break
                if self.is_ring(p):
                    # ring prevents anything further along the ray
                    break

        return free + rest

    def ring_moves(self, coordinate):
        moves = []
        for direction in directions:
            moves += self.ring_moves_in_direction(coordinate, direction)
        return moves

    def check_run(self, start_coord, direction, player):
        run = []
        for i in range(5):
            coord = add(start_coord, mul(direction, i))
            element = self.get_element(coord)
            if element is None or element[0] == "ring" or element[1] != player:
                return False, []
            else:
                run.append(coord)
        return True, run

    def has_runs(self, player):
        runs = []
        for direction, start_coords in zip([N, NE, SE], run_start_coords):
            check_coords = start_coords.intersection(set(self.board.markers[player]))
            for start_coord in check_coords:
                is_run, run = self.check_run(start_coord, direction, player)
                if is_run:
                    runs.append(run)
        return len(runs) > 0, runs

    def flip_markers_along_line(self, start, end):
        coords = get_points_between(start, end)
        for coord in coords:
            element = self.get_element(coord)
            if element is not None and self.is_marker(coord):
                self.board.modify_element(coord, ("marker", 1 - element[1]))

    def remove_run(self, run):
        for marker_coord in run:
            self.board.remove_element(marker_coord)

    def is_terminal(self) -> bool:
        """Check if game ends due to points or insufficienct markers to add"""
        return max(self.points) == self.points_to_win or (
            self.turn_type == "add marker"
            and len(self.get_markers(0) + self.get_markers(1)) >= MAX_MARKERS
        )

    def run_clearing_setup(self, runs):
        if len(runs) == 1:
            # remove the single run
            self.remove_run(runs[0])
            self.turn_type = "remove ring"
            self.valid_moves = self.get_rings(self.active_player)
        else:
            self.turn_type = "remove run"
            self.valid_moves = runs

    def set_valid_add_markers(self):
        self.valid_moves = [
            r
            for r in self.board.rings[self.active_player]
            if len(self.ring_moves(r)) > 0
        ]

    def setup_next_turn_and_player(self):
        if self.turn_type == "setup new rings":
            self.active_player = 1 - self.active_player
            if len(self.get_rings(1)) < 5:
                # still placing new rings
                self.valid_moves = [c for c in coords if self.is_free(c)]
            else:
                self.turn_type = "add marker"
                self.set_valid_add_markers()
        elif self.turn_type == "add marker":
            self.turn_type = "move ring"
            self.valid_moves = self.ring_moves(self.get_markers(self.active_player)[-1])
        elif self.turn_type == "move ring":
            has_a_run, runs = self.has_runs(self.active_player)
            if has_a_run:
                self.run_clearing_setup(runs)
            else:
                self.active_player = 1 - self.active_player
                has_a_run, runs = self.has_runs(self.active_player)
                if has_a_run:
                    self.run_clearing_setup(runs)
                else:
                    self.turn_type = "add marker"
                    self.set_valid_add_markers()
        elif self.turn_type == "remove run":
            self.turn_type = "remove ring"
            self.valid_moves = self.get_rings(self.active_player)
        elif self.turn_type == "remove ring":
            has_a_run, runs = self.has_runs(self.active_player)
            if has_a_run:
                self.run_clearing_setup(runs)
            else:
                if self.last_moved == self.active_player:
                    self.active_player = 1 - self.active_player
                    has_a_run, runs = self.has_runs(self.active_player)
                    if has_a_run:
                        self.run_clearing_setup(runs)
                    else:
                        self.turn_type = "add marker"
                        self.set_valid_add_markers()
                else:
                    self.turn_type = "add marker"
                    self.set_valid_add_markers()
        self.terminal = self.is_terminal()
        self.valid_moves.sort()
        return

    def take_turn(self, move):
        # Doesn't check whether move is valid.
        # Whatever interactions sends the move should check that
        # or have that guarantee somehow
        if self.turn_type == "setup new rings":
            # move assumed to be single coord of free space
            self.board.add_element(move, ("ring", self.active_player))
            # could record moves here, but I think I want that to be external as part of the interaction, rather than built into the bot.
        elif self.turn_type == "add marker":
            # move assumed to be single coord of active player's ring
            self.board.modify_element(move, ("marker", self.active_player))
            self.prev_ring = move
        elif self.turn_type == "move ring":
            # move is a valid position for the ring to move which was just replaced by a marker
            self.last_moved = self.active_player
            self.board.add_element(move, ("ring", self.active_player))
            self.flip_markers_along_line(self.prev_ring, move)
        elif self.turn_type == "remove run":
            # move assumed to be a lsit of five coords in a row
            self.remove_run(move)
        elif self.turn_type == "remove ring":
            # move is a ring of the active player
            self.points[self.active_player] += 1
            self.board.remove_element(move)
        self.setup_next_turn_and_player()

    @staticmethod
    def get_next_game_state(game_state, move):
        y = YinshGame(game_state)
        y.take_turn(move)
        return y.get_game_state()
