from functools import cache, lru_cache
import logging
from typing import Iterable, Tuple

from numpy.linalg import matrix_power
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from yinsh import GameState, GameBoard


log = logging.getLogger(__name__)


## coordinate stuff
def valid_coord(x, y):
    return (0.5 * np.sqrt(3) * x) ** 2 + (0.5 * x - y) ** 2 <= 4.6**2


COORDS = [(x, y) for x in range(-5, 6) for y in range(-5, 6) if valid_coord(x, y)]

SIGNED_BOARD_TENSOR_SIZE = 2 * len(
    COORDS
)  # copy of board for p1, p2 markers, p1,p2 rings
BOARD_TENSOR_SIZE = 4 * len(COORDS)  # copy of board for p1, p2 markers, p1,p2 rings
# if 2d, would be 11 x 11


def yinsh_exp_adjacency_tensor():
    def adjacent(xy, xy2):
        return (xy[0] - xy2[0]) ** 2 + (xy[1] - xy2[1]) ** 2 == 1 or (
            (xy[0] - xy2[0]) == 1 and (xy[1] - xy2[1]) == 1
        )

    mat = np.array([[1.0 if adjacent(p, q) else 0.0 for q in COORDS] for p in COORDS])

    return torch.matrix_exp(torch.Tensor(mat)), torch.Tensor(mat)  # exp adj, adjacency


@cache
def rot_60deg_clockwise(coord):
    """Multiply coord vec by
    [[1,  1]
    ,[-1, 0]] on right.
    """
    return ((coord[0] - coord[1]), coord[0])


@cache
def reflect_over_y(coord):
    """Multiply coord vec by
    [[-1, -1]
    ,[0,  1]] on right.
    """
    return (-coord[0], coord[1] - coord[0])


@cache
def apply_coord_transform(coord, rotations, reflection):
    tmp = coord
    for r in range(rotations):
        coord = rot_60deg_clockwise(coord)
    if reflection % 2:
        return reflect_over_y(coord)
    # assert valid_coord(
    #     *coord
    # ), f"invalid coord transform!! c:{tmp}, rot: {rotations}, ref: {reflection} -> {coord}"
    return coord


@cache
def transform_matrix(rotations, reflections):
    rot = np.array([[1, 1], [-1, 0]])
    ref = np.array([[-1, -1], [0, 1]])
    return matrix_power(ref, reflections) @ matrix_power(rot, rotations)


# @lru_cache(maxsize=12 * 3 * 100)
def transform_coord_list(coord_list, rotations, reflections, sort_coords=False):
    if coord_list:
        if sort_coords:
            return sorted(
                [
                    tuple(t)
                    for t in (
                        np.array(coord_list) @ transform_matrix(rotations, reflections)
                    )
                ]
            )
        return [
            tuple(t)
            for t in (np.array(coord_list) @ transform_matrix(rotations, reflections))
        ]
    return []


def board_symmetry(board, rotations, reflections, sort_coords=False):
    return GameBoard(
        elements={
            apply_coord_transform(coord, rotations, reflections): element
            for coord, element in board.elements.items()
        },
        rings=[
            transform_coord_list(board.rings[0], rotations, reflections, sort_coords),
            transform_coord_list(board.rings[1], rotations, reflections, sort_coords),
        ],
        markers=[
            transform_coord_list(board.markers[0], rotations, reflections, sort_coords),
            transform_coord_list(board.markers[1], rotations, reflections, sort_coords),
        ],
    )


def game_state_symmetry(game_state, rotations, reflections, sort_coords=False):
    if game_state.turn_type != "remove run":
        vm = transform_coord_list(
            game_state.valid_moves, rotations, reflections, sort_coords
        )
    else:
        vm = [
            sorted(
                transform_coord_list(m, rotations, reflections),
                key=lambda xy: xy[0] * 10 + xy[1],
            )
            for m in game_state.valid_moves
        ]
    return GameState(
        game_state.turn_type,
        game_state.active_player,
        board_symmetry(game_state.board, rotations, reflections, sort_coords),
        game_state.points,
        game_state.points_to_win,
        vm,
        game_state.last_moved,
        apply_coord_transform(game_state.prev_ring, rotations, reflections),
        game_state.max_markers_before_draw,
        game_state.terminal,
        game_state.turn,
    )


def get_symmetries(game_state, history, probabilities):
    """Return a list of equivalent game_state, probability combos.

    It's a hexagon, so the symmetries of the game are the dihedral group of order 12.
    A rotation of order 6 and a reflection are the generators.
    """
    symmetries = []
    for rot in range(6):
        for ref in range(2):
            new_state = game_state_symmetry(game_state, rot, ref)
            new_history = [board_symmetry(h, rot, ref) for h in history]
            # re-order moves and probabilities
            moves = new_state.valid_moves
            sort_inds = sorted(
                range(len(moves)), key=lambda i: moves[i]
            )  # I hate that we have to sort this every time, but without re-working entirely how valid moevs works, I think it's necessary to make probs work correctly
            new_prob = np.array(
                [probabilities[sort_inds[i]] for i in range(len(sort_inds))]
            )
            new_moves = [moves[sort_inds[i]] for i in range(len(sort_inds))]
            new_state.valid_moves = new_moves
            symmetries.append((new_state, new_history, new_prob))
    return symmetries


def get_state_symmetries(game_state):
    """Return a list of equivalent game_state

    It's a hexagon, so the symmetries of the game are the dihedral group of order 12.
    A rotation of order 6 and a reflection are the generators.
    """
    symmetries = []
    for rot in range(6):
        for ref in range(2):
            symmetries.append(game_state_symmetry(game_state, rot, ref, True))
    return symmetries


def binary_tensor_from_list(keys, index_dict) -> torch.Tensor:
    arr_01 = torch.zeros((len(index_dict),))
    for key in keys:
        arr_01[index_dict[key]] = 1
    return arr_01


def value_tensor_from_list(keys, values, index_dict) -> torch.Tensor:
    arr = torch.zeros((len(index_dict),))
    for key, value in zip(keys, values):
        if isinstance(value, np.ndarray):
            # print(type(value))
            print(value)
            print(values)
            print(keys)
        arr[index_dict[key]] = float(value)
    return arr


class PreprocessGamestateFlat:
    """Encode the gamestate as a flattened (2 x 85 x T) + 11 dimensional tensor.

    example (T, input dim): (1, 185), (2, 359), (3, 533), (4, 707), (5, 881), (6, 1055), (7, 1229), (8, 1403)
    This tensor will be the input to our NN-based yinsh bots.
    T is the number of current and previous boards used as input to the model.
    T=1 only returns the current board with no past boards.
    Each board is concatenation of 4 binary encodings of size 85 (number of board spaces)
    Could make this an over-full image by making it 11 x 11 and just adding 5 to every coordinate and literally stacking images, but we haven't yet.
    -> player markers - opponent markers
    -> player rings - opponent rings
    The 11 additional dimensions include
    -> 5 dimensional one hot encoding for the turn type
    -> \pm1 for current player was last to move
    -> player's points: int
    -> opponents points: int
    -> points required to win. This is always constant for a given game variant, usually 3
    -> number of markers on the board = sum (of first two board planes)
    -> number of total markers. If this many markers are on board and the turn type is add marker, game ends.
    """

    def __init__(
        self,
        history_length: int = 8,  # If game is fully observable, not needed. Yinsh needs to know which ring had a marker added when moving a ring. what happened last if we were removing a ring in order to know who goes after our move, etc.
    ) -> None:
        self.coord_inds = {coord: i for i, coord in enumerate(COORDS)}
        self.turn_inds = {
            "setup new rings": 0,
            "add marker": 1,
            "move ring": 2,
            "remove run": 3,
            "remove ring": 4,
        }
        self.history_length = history_length
        # self.history = torch.zeros(
        #     (BOARD_TENSOR_SIZE * self.history_length,)
        # )  # board history

    def board_masks(self, board, pr) -> Tuple:
        m1 = binary_tensor_from_list(board.markers[0], self.coord_inds)
        m2 = binary_tensor_from_list(board.markers[1], self.coord_inds)
        r1 = binary_tensor_from_list(board.rings[0] + pr[0], self.coord_inds)
        r2 = binary_tensor_from_list(board.rings[1] + pr[1], self.coord_inds)
        return m1 - m2, r1 - r2

    def turn_mask(self, turn_type) -> torch.Tensor:
        return binary_tensor_from_list([turn_type], self.turn_inds)

    def points_tensor(self, points, player) -> torch.Tensor:
        """active palyer perspective points"""
        return torch.Tensor([points[player], points[1 - player]])

    def game_state_to_tensors(self, gs: GameState) -> Tuple:
        pr = [[], []]
        if gs.turn_type == "move ring":
            pr[gs.active_player] = [gs.prev_ring]

        board = torch.cat(self.board_masks(gs.board, pr))  # (85 * 2)-D

        turn_type = self.turn_mask(gs.turn_type)  # 5-D
        last_moved = torch.Tensor([[1, -1][gs.active_player == gs.last_moved]])  # 1-D
        points = self.points_tensor(gs.points, gs.active_player)  # 2-D

        win_points = torch.Tensor([gs.points_to_win])  # 1-D
        curr_markers = torch.Tensor([sum([len(m) for m in gs.board.markers])])  # 1-D
        max_markers = torch.Tensor([gs.max_markers_before_draw])  # 1-D
        # gs.valid_moves # might help, but might hurt... not sure. should probably test both  # TODO
        # gs.prev_ring, # see without these due to history
        # gs.terminal, # bot won't move if terminal so ignore this
        return (
            board,
            turn_type,
            last_moved,
            points,
            win_points,
            curr_markers,
            max_markers,
        )

    def update_history(current_board):
        self.history[:-BOARD_TENSOR_SIZE] = self.history[BOARD_TENSOR_SIZE:]
        self.history[-BOARD_TENSOR_SIZE:] = current_board

    def full_tensor(
        self, gs: GameState, board_history: Iterable[GameBoard]
    ) -> torch.Tensor:
        (
            board,
            turn_type,
            last_moved,
            points,
            win_points,
            curr_markers,
            max_markers,
        ) = self.game_state_to_tensors(gs)
        if self.history_length > 1:
            history = torch.cat(
                [torch.cat(self.board_masks(b)) for b in board_history] + [board]
            )
        else:
            history = board
        # print(
        #     f"\nDEBUG H:{history} TT:{turn_type} PP:{points} WP:{win_points} CM:{curr_markers} MM:{max_markers}\n"
        # )
        # print(
        #     f"\nDEBUG H:{history.size()} TT:{turn_type.size()} PP:{points.size()} WP:{win_points.size()} CM:{curr_markers.size()} MM:{max_markers.size()}\n"
        # )
        return torch.cat(
            (
                [1, -1][gs.active_player] * history,
                turn_type,
                last_moved,
                points,
                win_points,
                curr_markers,
                max_markers,
            )
        )


class PostprocessMoveDistributionFlat:
    """Decode probability distribution Tensor and GameState to probability over valid moves.

    The distribution is over a single board dim = (85,) to say which piece to move.
    When removing runs, the sum of coordinate probabilities of each marker in a run is used.
    """

    def __init__(self):
        self.coord_inds = {coord: i for i, coord in enumerate(COORDS)}

    def valid_move_distribution(self, model_probs, valid_moves) -> np.array:
        # convert model distribution over all values into distribution over valid moves
        #  DEBUG there's probably some post-processing shennanigans I need to work out with typing from tensors to output
        probs = np.zeros((len(valid_moves),))
        if type(valid_moves[0]) != list:
            for i, move in enumerate(valid_moves):
                probs[i] = model_probs[self.coord_inds[move]]
        else:
            for i, move in enumerate(valid_moves):
                for marker in move:
                    probs[i] += model_probs[self.coord_inds[marker]]
        # better not be all zeros or have NaNs here >:C
        s = np.sum(probs)
        if s > 0:
            probs = probs / np.sum(probs)
        else:
            log.error("All valid moves were masked, doing a workaround.")
            probs = np.array([1 / len(valid_moves) for _ in valid_moves])
        return probs

    def to_policy(self, move_probs, valid_moves) -> np.array:
        # convert model distribution over all values into distribution over valid moves
        #  DEBUG there's probably some post-processing shennanigans I need to work out with typing from tensors to output
        if type(valid_moves[0]) != list:
            return value_tensor_from_list(valid_moves, move_probs, self.coord_inds)
        # remove run shennanigans
        coords = []
        probs = []
        for prob, move in zip(move_probs, valid_moves):
            coords += move
            probs += [prob] * 5
        probs = np.array(probs) / sum(probs)
        return value_tensor_from_list(coords, probs, self.coord_inds)


class YinshNetFlat(nn.Module):
    def __init__(self, args):
        super(YinshNetFlat, self).__init__()
        self.preprocessing = PreprocessGamestateFlat(args.history_length)
        self.postprocessing = PostprocessMoveDistributionFlat()
        self.input_size = self.preprocessing.full_tensor(
            GameState(), [GameBoard() for _ in range(args.history_length)]
        ).numel()
        self.action_size = len(COORDS)

        if "internal_width" not in args:
            internal_width = int(self.input_size)
        else:
            internal_width = args.internal_width

        self.fc_layers = nn.Sequential(
            nn.Linear(self.input_size, internal_width),
            nn.BatchNorm1d(internal_width),
            nn.ReLU(),
            *[
                nn.Sequential(
                    nn.Linear(internal_width, internal_width),
                    nn.BatchNorm1d(internal_width),
                    nn.ReLU(),
                    nn.Dropout(args.dropout),
                )
                for _ in range(args.internal_layers)
            ],
            nn.Linear(internal_width, args.external_width),
            nn.BatchNorm1d(args.external_width),
            nn.ReLU(),
            nn.Dropout(args.dropout),
        )

        self.fc_policy = nn.Linear(args.external_width, self.action_size)
        self.probify = nn.Softmax(dim=-1)  # squish into (0,1)

        self.fc_value = nn.Linear(args.external_width, 1)
        self.signify = nn.Tanh()  # squish into (-1,1)

    def preprocess(self, game_state: GameState, history):
        """Convert gamestate to a tensor"""
        return self.preprocessing.full_tensor(game_state, history)

    def postprocess(self, policy, valid_moves):
        return self.postprocessing.valid_move_distribution(policy, valid_moves)

    def forward(self, state: torch.Tensor):
        state = state.view(-1, self.input_size)
        state = self.fc_layers(state)
        policy = self.fc_policy(state)
        value = self.fc_value(state)
        return self.probify(policy), self.signify(value)


class GRes(nn.Module):
    """Implement Module of GResNet Graph raw from below paper with batchnorm and dropout
    https://arxiv.org/pdf/1909.05729.pdf
    """

    def __init__(self, in_width, out_width, dropout=0):
        super(GRes, self).__init__()
        # self.filter = filter  # having this here would mean a ton of duplicated memory, so pass in x prefiltered instead
        self.lin = nn.Linear(in_width, out_width)
        self.bn = nn.BatchNorm1d(out_width)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, filtered_x, residual=None):
        if residual is not None:
            return self.dropout(self.relu(residual + self.bn(self.lin(filtered_x))))
        return self.dropout(self.relu(self.bn(self.lin(filtered_x))))


class YinshNetGraph(nn.Module):
    """implement a graph resnet on the yinsh board"""

    def __init__(self, args):
        super(YinshNetGraph, self).__init__()
        self.preprocessing = PreprocessGamestateFlat(args.history_length)
        self.postprocessing = PostprocessMoveDistributionFlat()
        self.input_size = self.preprocessing.full_tensor(GameState(), []).numel()
        self.action_size = len(COORDS)
        internal_width = int(self.input_size)
        external_width = int(self.input_size) - len(COORDS)
        # network structure
        self.fc_layers_double = nn.ModuleList(
            [
                GRes(
                    self.input_size,
                    internal_width,
                    args.dropout,
                )
            ]
            + [
                GRes(
                    internal_width,
                    internal_width,
                    args.dropout,
                )
                for _ in range(args.internal_layers_doubled - 1)
            ]
        )
        self.transition = GRes(
            internal_width,
            external_width,
            args.dropout,
        )
        self.fc_layers_single = nn.ModuleList(
            [
                GRes(
                    external_width,
                    external_width,
                    args.dropout,
                )
                for _ in range(args.internal_layers_singled)
            ]
        )
        # Graph convolution matrices
        exp_adj, adj = yinsh_exp_adjacency_tensor()

        if args.filter == "exp":
            mat = exp_adj
        else:
            mat = adj + torch.eye(len(COORDS))
        mat = torch.cat((mat, mat), dim=0)
        mat = torch.cat((mat, mat), dim=1)
        D_minus_half = torch.diagflat(1 / torch.sqrt(torch.sum(mat, dim=1)))
        D_minus_half_single = torch.diagflat(
            1 / torch.sqrt(torch.sum(mat[: len(COORDS), : len(COORDS)], dim=1))
        )

        # normalized graph convolution filter from https://arxiv.org/pdf/1609.02907.pdf
        self.filter_double_graph = torch.eye(max([self.input_size, internal_width]))
        self.filter_double_graph[: len(COORDS) * 2, : len(COORDS) * 2] = (
            D_minus_half @ mat @ D_minus_half
        ).t()
        self.filter_double_graph = self.filter_double_graph.to(torch.device(args.gpu))
        # normalized graph convolution filter from https://arxiv.org/pdf/1609.02907.pdf
        self.filter_single_graph = torch.eye(external_width)
        self.filter_single_graph[: len(COORDS), : len(COORDS)] = (
            D_minus_half_single
            @ mat[: len(COORDS), : len(COORDS)]
            @ D_minus_half_single
        ).t()
        self.filter_single_graph = self.filter_single_graph.to(torch.device(args.gpu))
        self.fc_policy = nn.Linear(external_width, self.action_size)
        self.fc_valid = nn.Linear(external_width, self.action_size)
        self.probify = nn.Softmax(dim=-1)  # squish into (0,1)

        self.maskify = nn.Sigmoid()  # squish into (0,1)

        self.fc_value = nn.Linear(external_width, 1)
        self.signify = nn.Tanh()  # squish into (-1,1)

    def preprocess(self, game_state: GameState, history):
        """Convert gamestate to a tensor"""
        return self.preprocessing.full_tensor(game_state, history)

    def postprocess(self, policy, valid_moves):
        return self.postprocessing.valid_move_distribution(policy, valid_moves)

    def forward(self, state: torch.Tensor):
        state = state.view(-1, self.input_size)
        residual_double = torch.matmul(state, self.filter_double_graph)
        s_state = torch.clone(state[:, len(COORDS) :])
        s_state[:, : len(COORDS)] += state[:, : len(COORDS)]

        residual_single = torch.matmul(s_state, self.filter_single_graph)
        for layer in self.fc_layers_double:
            state = layer(
                torch.matmul(state, self.filter_double_graph), residual_double
            )
        state = self.transition(
            torch.matmul(state, self.filter_double_graph), residual_single
        )
        for layer in self.fc_layers_single:
            state = layer(
                torch.matmul(state, self.filter_single_graph), residual_single
            )
        policy = self.fc_policy(state)
        valid = self.maskify(self.fc_valid(state))
        value = self.fc_value(state)
        return self.probify(policy * valid), self.signify(value), valid

    # def train(self, examples):
    #     """
    #     This function trains the neural network with examples obtained from
    #     self-play.

    #     Input:
    #         examples: a list of training examples, where each example is of form
    #                   (board, pi, v). pi is the MCTS informed policy vector for
    #                   the given board, and v is its value. The examples has
    #                   board in its canonical form.
    #     """
    #     pass

    # def predict(self, board):
    #     """
    #     Input:
    #         board: current board in its canonical form.

    #     Returns:
    #         pi: a policy vector for the current board- a numpy array of length
    #             game.getActionSize
    #         v: a float in [-1,1] that gives the value of the current board
    #     """
    #     pass

    # def save_checkpoint(self, folder, filename):
    #     """
    #     Saves the current neural network (with its parameters) in
    #     folder/filename
    #     """
    #     pass

    # def load_checkpoint(self, folder, filename):
    #     """
    #     Loads parameters of the neural network from folder/filename
    #     """
    #     pass

    # def train(self, examples):
    #     """
    #     This function trains the neural network with examples obtained from
    #     self-play.

    #     Input:
    #         examples: a list of training examples, where each example is of form
    #                   (board, pi, v). pi is the MCTS informed policy vector for
    #                   the given board, and v is its value. The examples has
    #                   board in its canonical form.
    #     """
    #     pass

    # def predict(self, board):
    #     """
    #     Input:
    #         board: current board in its canonical form.

    #     Returns:
    #         pi: a policy vector for the current board- a numpy array of length
    #             game.getActionSize
    #         v: a float in [-1,1] that gives the value of the current board
    #     """
    #     pass

    # def save_checkpoint(self, folder, filename):
    #     """
    #     Saves the current neural network (with its parameters) in
    #     folder/filename
    #     """
    #     pass

    # def load_checkpoint(self, folder, filename):
    #     """
    #     Loads parameters of the neural network from folder/filename
    #     """
    #     pass


# class ValuePolicyLoss(nn.Module):
#     def __init__(self, l2_reg: float):
#         self.sq_error = nn.MSELoss()
#         self.ce_loss = nn.CrossEntropyLoss()

#     def forward(value, policy, target_value, target_policy):
#         return self.sq_error(value, target_value) + self.ce_loss(policy, target_policy)


# class PreprocessGamestate2D:
#     # TODO: This is still the old flat one
#     """Encode the gamestate as a flattened (4 x 85 x T) + 10 dimensional tensor.

#     total input dim: T=8: 2794, T=4: 1402, T=3: 1054, T=2: 706, T=1: 358
#     This tensor will be the input to our NN-based yinsh bots.
#     T is the number of current and previous boards used as input to the model.
#     T=1 only returns the current board with no past boards.
#     Each board is concatenation of 4 binary encodings of size 85 (number of board spaces)
#     Could make this an over-full image by making it 11 x 11 and just adding 5 to every coordinate and literally stacking images, but we haven't yet.
#     That would add a ton of useless dimensionality since 121 >> 85 and all of those extra dimensions would be masked out every time.
#     -> player markers
#     -> opponent markers
#     -> player rings
#     -> opponent rings
#     The 10 additional dimensions include
#     -> 5 dimensional one hot encoding for the turn type
#     -> player's points: int
#     -> opponents points: int
#     -> points required to win. This is always constant for a given game variant, usually 3
#     -> number of markers on the board = sum (of first two board planes)
#     -> number of total markers. If this many markers are on board and the turn type is add marker, game ends.
#     """

#     def __init__(
#         self,
#         history_length: int = 8,  # If game is fully observable, not needed. Yinsh needs to know which ring had a marker added when moving a ring. what happened last if we were removing a ring in order to know who goes after our move, etc.
#         augment_input: bool = True,
#     ) -> None:
#         self.coord_inds = {coord: i for i, coord in enumerate(COORDS)}
#         self.turn_inds = {
#             "setup new rings": 0,
#             "add marker": 1,
#             "move ring": 2,
#             "remove run": 3,
#             "remove ring": 4,
#         }
#         self.history_length = history_length
#         self.history = [
#             torch.zeros((BOARD_TENSOR_SIZE * self.history_length,)) for _ in range(2)
#         ]  # board history

#     def board_masks(self, board, player) -> Tuple:
#         m1 = binary_tensor_from_list(board.markers[player], self.coord_inds)
#         m2 = binary_tensor_from_list(board.markers[1 - player], self.coord_inds)
#         r1 = binary_tensor_from_list(board.rings[player], self.coord_inds)
#         r2 = binary_tensor_from_list(board.rings[1 - player], self.coord_inds)
#         return m1, m2, r1, r2

#     def turn_mask(self, turn_type) -> torch.Tensor:
#         return binary_tensor_from_list([turn_type], self.turn_inds)

#     def points_tensor(self, points, player) -> torch.Tensor:
#         return torch.Tensor([points[player], points[1 - player]])

#     def game_state_to_tensors(self, gs: GameState) -> Tuple:
#         board0 = torch.cat(board_masks(gs.board, 0))  # (85 * 4)-D
#         board1 = torch.cat(
#             board_masks(gs.board, 1)
#         )  # only one of the two boards is used
#         turn_type = turn_mask(gs.turn_type)  # 5-D
#         points = points_tensor(gs.points, gs.active_player)  # 2-D

#         win_points = torch.Tensor([gs.points_to_win])  # 1-D
#         curr_markers = torch.Tensor([sum([len(m) for m in gs.board.markers])])  # 1-D
#         max_markers = torch.Tensor([gs.max_markers_before_draw])  # 1-D

#         # gs.last_moved, # see without these due to history
#         # gs.prev_ring, # see without these due to history
#         # gs.terminal, # bot won't move if terminal so ignore this
#         return board, turn_type, points, win_points, curr_markers, max_markers

#     def update_history(current_board, i):
#         self.history[player][:-BOARD_TENSOR_SIZE] = self.history[player][
#             BOARD_TENSOR_SIZE:
#         ]
#         self.history[player][-BOARD_TENSOR_SIZE:] = current_board

#     def get_input_tensor(self, gs: GameState) -> torch.Tensor:
#         (
#             board0,
#             board1,
#             turn_type,
#             points,
#             win_points,
#             curr_markers,
#             max_markers,
#         ) = self.game_state_to_tensors(gs)
#         self.update_history(board0, board1)
#         return torch.cat(
#             (self.history, turn_type, points, win_points, curr_markers, max_markers)
#         )
