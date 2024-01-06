from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from yinsh import GameState


## coordinate stuff
def valid_coord(x, y):
    return (0.5 * np.sqrt(3) * x) ** 2 + (0.5 * x - y) ** 2 <= 4.6**2


COORDS = [(x, y) for x in range(-5, 6) for y in range(-5, 6) if valid_coord(x, y)]

SIGNED_BOARD_TENSOR_SIZE = 2 * len(
    COORDS
)  # copy of board for p1, p2 markers, p1,p2 rings
BOARD_TENSOR_SIZE = 4 * len(COORDS)  # copy of board for p1, p2 markers, p1,p2 rings
# if 2d, would be 11 x 11


def binary_tensor_from_list(key_list, index_dict) -> torch.Tensor:
    arr_01 = torch.zeros((len(index_dict),))
    for key in key_list:
        arr_01[index_dict[val]] = 1
    return arr_01


def rot_60deg_clockwise(coord):
    """multiply coord vec by [[0 1], [-1 1]]"""
    return (coord[1], coord[1] - coord[0])


def reflect_over_y(coord):
    """multiply coord vec by [[-1 0], [-1 1]]"""
    return (-coord[0], coord[1] - coord[0])


def augment_board(board: GameBoard, augmentations: Iterable[Callable]) -> GameBoard:
    return GameBoard(
        elements={
            a(coord): element
            for a in augmentations
            for coord, element in board.elements
        },
        rings=[[a(m) for a in augmentations for r in rs] for rs in board.rings],
        markers=[[a(m) for a in augmentations for m in ms] for ms in board.markers],
    )


class PreprocessGamestateFlat:
    """Encode the gamestate as a flattened (2 x 87 x T) + 10 dimensional tensor.

    example (T, input dim): (1, 184), (2, 358), (3, 532), (4, 706), (5, 880), (6, 1054), (7, 1228), (8, 1402)
    This tensor will be the input to our NN-based yinsh bots.
    T is the number of current and previous boards used as input to the model.
    T=1 only returns the current board with no past boards.
    Each board is concatenation of 4 binary encodings of size 87 (number of board spaces)
    Could make this an over-full image by making it 11 x 11 and just adding 5 to every coordinate and literally stacking images, but we haven't yet.
    -> player markers - opponent markers
    -> player rings - opponent rings
    The 10 additional dimensions include
    -> 5 dimensional one hot encoding for the turn type
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
        self.history = torch.zeros(
            (BOARD_TENSOR_SIZE * self.history_length,)
        )  # board history

    def board_masks(self, board, player) -> Tuple:
        m1 = binary_tensor_from_list(board.markers[0], self.coord_inds)
        m2 = binary_tensor_from_list(board.markers[1], self.coord_inds)
        r1 = binary_tensor_from_list(board.rings[0], self.coord_inds)
        r2 = binary_tensor_from_list(board.rings[1], self.coord_inds)
        return m1 - m2, r1 - r2

    def turn_mask(self, turn_type) -> torch.Tensor:
        return binary_tensor_from_list([turn_type], self.turn_inds)

    def points_tensor(self, points, player) -> torch.Tensor:
        return torch.Tensor([points[player], points[1 - player]])

    def game_state_to_tensors(self, gs: GameState) -> Tuple:
        board0 = torch.cat(board_masks(gs.board))  # (87 * 4)-D
        turn_type = turn_mask(gs.turn_type)  # 5-D
        points = points_tensor(gs.points, gs.active_player)  # 2-D

        win_points = torch.Tensor([gs.points_to_win])  # 1-D
        curr_markers = torch.Tensor([sum([len(m) for m in gs.board.markers])])  # 1-D
        max_markers = torch.Tensor([gs.max_markers_before_draw])  # 1-D
        # gs.valid_moves # might help, but might hurt... not sure. should probably test both  # TODO
        # gs.last_moved, # see without these due to history
        # gs.prev_ring, # see without these due to historu
        # gs.terminal, # bot won't move if terminal so ignore this
        return board, turn_type, points, win_points, curr_markers, max_markers

    def update_history(current_board):
        self.history[player][:-BOARD_TENSOR_SIZE] = self.history[player][
            BOARD_TENSOR_SIZE:
        ]
        self.history[player][-BOARD_TENSOR_SIZE:] = current_board

    def get_input_tensor(self, gs: GameState) -> torch.Tensor:
        (
            board,
            turn_type,
            points,
            win_points,
            curr_markers,
            max_markers,
        ) = self.game_state_to_tensors(gs)

        self.update_history(board)

        return torch.cat(
            (
                [1, -1][gs.active_player] * self.history,
                turn_type,
                points[gs.active_player],
                points[1 - gs.active_player],
                win_points,
                curr_markers,
                max_markers,
            )
        )


class PostprocessMoveDistributionFlat:
    """Decode probability distribution Tensor and GameState to probability over valid moves.

    The distribution is over a single board dim = (87,) to say which piece to move.
    When removing runs, the sum of coordinate probabilities of each marker in a run is used.
    """

    def __init__(self):
        self.coord_inds = {coord: i for i, coord in enumerate(COORDS)}

    def valid_move_distribution(model_probs, valid_moves):
        # convert model distribution over all values into distribution over valid moves
        #  DEBUG there's probably some post-processing shennanigans I need to work out with typing from tensors to output
        probs = np.zeros((len(valid_moves),))
        if type(valid_moves[0]) != list:
            for i, move in enumerate(valid_moves):
                probs[i] = model_prob[coord_inds[move]]
        else:
            for i, move in enumerate(valid_moves):
                for marker in move:
                    probs[i] += sum(model_prob[coord_inds[marker]])
        # better not be all zeros or have NaNs here >:C
        probs = probs / np.sum(probs)
        return probs


class YinshNetFlat(nn.module):
    def __init__(self, history_length: int, internal_layers: int, external_width: int):
        super(YinshNetFlat, self).__init__()
        self.preprocessing = PreprocessGamestateFlat(history_length)
        self.postprocessing = PostprocessMoveDistributionFlat()
        self.input_size = (SIGNED_BOARD_TENSOR_SIZE * history_length) + 10
        self.output_size = len(COORDS)

        internal_width = int(self.input_size)
        external_width
        self.fc_layers = nn.Sequential(
            nn.Linear(self.input_size, internal_width),
            nn.BatchNorm1d(wid1),
            nn.ReLU(),
            *[
                nn.Sequential(
                    nn.Linear(internal_wid, wid1),
                    nn.BatchNorm1d(wid1),
                    nn.ReLU(),
                    nn.Dropout(),
                )
                for _ in range(internal_layers)
            ],
            nn.Linear(internal_width, external_width),
            nn.BatchNorm1d(wid1),
            nn.ReLU(),
            nn.Dropout()
        )

        self.fc_policy = nn.Linear(external_width, self.action_size)
        self.probify = nn.Softmax()  # squish into (0,1)

        self.fc_value = nn.Linear(external_width, 1)
        self.signify = nn.Tanh()  # squish into (-1,1)

    def preprocess(self, game_state: GameState):
        return self.pre_processing(game_state)  # updates state history as well

    def postprocess(self, policy, valid_moves):
        return self.pre_processing(policy, valid_moves)

    def forward(self, state: Tensor):
        state = state.view(-1, 1, self.input_size)
        state = self.fc_layers(state)
        policy = self.fc_policy(state)
        value = self.fc_value(state)
        return self.probify(policy), self.signify.tanh(value)

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


class ValuePolicyLoss(nn.module):
    def __init__(self, l2_reg: float):
        self.sq_error = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(value, policy, target_value, target_policy):
        return self.sq_error(value, target_value) + self.ce_loss(policy, target_policy)


# class PreprocessGamestate2D:
#     # TODO: This is still the old flat one
#     """Encode the gamestate as a flattened (4 x 87 x T) + 10 dimensional tensor.

#     total input dim: T=8: 2794, T=4: 1402, T=3: 1054, T=2: 706, T=1: 358
#     This tensor will be the input to our NN-based yinsh bots.
#     T is the number of current and previous boards used as input to the model.
#     T=1 only returns the current board with no past boards.
#     Each board is concatenation of 4 binary encodings of size 87 (number of board spaces)
#     Could make this an over-full image by making it 11 x 11 and just adding 5 to every coordinate and literally stacking images, but we haven't yet.
#     That would add a ton of useless dimensionality since 121 >> 87 and all of those extra dimensions would be masked out every time.
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
#         board0 = torch.cat(board_masks(gs.board, 0))  # (87 * 4)-D
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
