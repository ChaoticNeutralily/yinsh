import logging
import os
import sys
from collections import deque
from copy import deepcopy
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from yinsh import GameState, GameBoard
from alphazero.nn_utils import get_symmetries
from alphazero.mcts import MCTS

log = logging.getLogger(__name__)


def play_game(make_move1, make_move2, yinsh_game, history_len):
    gs = yinsh_game.get_game_state()
    history = [GameBoard() for _ in range(history_len - 1)]
    # print(f"DEBUG: history = {history}")
    while not gs.terminal:
        move = [make_move1, make_move2][gs.active_player](gs, history)
        # assumed to be valid
        if history:
            history.append(gs.board)
            history = history[1:]

        yinsh_game.take_turn(move)
        gs = yinsh_game.get_game_state()

    if gs.points[0] > gs.points[1]:
        return np.array([1, 0, 0])
    elif gs.points[1] > gs.points[0]:
        return np.array([0, 1, 0])
    return np.array([0, 0, 1])


def final_value(end, points, player):
    """+1 if player won, -1 if other player won, and 0 if not ended, eps if draw."""
    if not end:
        return 0
    elif points[player] > points[1 - player]:
        return 1
    elif points[player] < points[1 - player]:
        return -1
    # draw
    return 1e-4


class Coach:
    """
    Directly adapted from https://github.com/suragnair/alpha-zero-general/blob/master/Coach.py
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(
            self.nnet.nnet.__class__, self.nnet.args
        )  # the competitor network, new instance of same class
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.train_examples_history = (
            []
        )  # history of examples from args.numItersFortrain_examples_history latest iterations
        self.skip_first_self_play = False  # can be overriden in load_train_examples()
        self.rng = np.random.default_rng()

    def execute_episode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        train_examples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in train_examples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            train_examples: a list of examples of the form (state, history, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player won the game, -1 if lost, 0 if draw.
        """
        train_examples = []
        game = self.game()
        self.cur_player = 0
        episode_step = 0
        history = [deepcopy(game.board) for _ in range(self.nnet.history_length - 1)]
        while True:
            episode_step += 1
            # canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episode_step < self.args.temp_threshold)

            game_state = game.get_game_state()

            probs = self.mcts.get_action_probability(
                game.get_game_state(), history, temp=temp
            )
            sym = get_symmetries(game_state, history, probs)
            for sym_state, sym_hist, sym_prob in sym:
                train_examples.append(
                    [
                        sym_state,
                        deepcopy(sym_hist),
                        self.cur_player,
                        self.nnet.to_policy(sym_prob, sym_state.valid_moves),
                    ]
                )
            game_state, history, probs = sym[0]
            valid_moves = game_state.valid_moves
            action = self.rng.choice(valid_moves, p=probs)
            if type(valid_moves[0]) == tuple:
                action = tuple(action)
            else:
                action = [tuple(coord) for coord in action]
            if history:
                history.append(deepcopy(game.board))
                history = history[1:]

            game.take_turn(action)

            self.cur_player = game.active_player

            # r is the current player's reward
            r = final_value(game.terminal, game.points, self.cur_player)

            if game.terminal:
                # return [(state, history, current_player, policy, value) for each example]
                if np.abs(r) < 1:
                    # return all with same value of 0 for draw
                    return [(x[0], x[1], x[3], 0) for x in train_examples]
                else:
                    return [
                        (x[0], x[1], x[3], r * [-1, 1][x[2] == self.cur_player])
                        for x in train_examples
                    ]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in train_examples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(self.args.first_iter, self.args.num_iters + 1):
            # bookkeeping
            log.info(f"Starting Iter #{i} ...")
            # examples of the iteration
            if not self.skip_first_self_play or i > 1:
                iteration_train_examples = deque([], maxlen=self.args.maxlen_of_queue)

                for _ in tqdm(range(self.args.num_episodes), desc="Self Play"):
                    self.mcts = MCTS(
                        self.game, self.nnet, self.args
                    )  # reset search tree
                    iteration_train_examples += self.execute_episode()
                log.info("Finished self play, Appending examples to training history.")
                # save the iteration examples to the history
                self.train_examples_history.append(iteration_train_examples)

            if (
                len(self.train_examples_history)
                > self.args.num_iters_for_train_examples_history
            ):
                log.warning(
                    f"Removing the oldest entry in train_examples. len(train_examples_history) = {len(self.train_examples_history)}"
                )
                self.train_examples_history.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            log.info(f"Saving training examples to checkpoint {i-1}.")
            self.save_train_examples(i - 1)

            # shuffle examples before training
            train_examples = []
            for e in self.train_examples_history:
                train_examples.extend(e)
            shuffle(train_examples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(
                folder=self.args.checkpoint, filename="temp.pth.tar"
            )
            self.pnet.load_checkpoint(
                folder=self.args.checkpoint, filename="temp.pth.tar"
            )
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(train_examples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info("PITTING AGAINST PREVIOUS VERSION")
            results = np.zeros((3,))
            for _ in tqdm(range(self.args.num_arena_games), desc=f"Playing yinsh game"):
                results += play_game(
                    pmcts.highest_prob_move,
                    nmcts.highest_prob_move,
                    self.game(GameState()),
                    self.nnet.history_length,
                )
            pwins, nwins, draws = results.astype(int)

            log.info(
                "NEW MODEL WINS / DRAWS / LOSSES : %d / %d / %d" % (nwins, draws, pwins)
            )
            if (
                (float(nwins) + 0.5 * float(draws))
                / self.args.num_arena_games  # average score
            ) < self.args.update_threshold:
                log.info("REJECTING NEW MODEL")
                self.nnet.load_checkpoint(
                    folder=self.args.checkpoint, filename="best.pth.tar"
                )
            else:
                log.info("ACCEPTING NEW MODEL")
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint, filename=self.get_checkpoint_file(i)
                )
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint, filename="best.pth.tar"
                )

    def get_checkpoint_file(self, iteration):
        return "checkpoint_" + str(iteration) + ".pth.tar"

    def save_train_examples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(
            folder, self.get_checkpoint_file(iteration) + ".examples"
        )
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.train_examples_history)
        f.closed

    def load_train_examples(self):
        model_file = os.path.join(
            self.args.load_folder_file[0], self.args.load_folder_file[1]
        )
        examples_file = model_file + ".examples"
        if not os.path.isfile(examples_file):
            log.warning(f'File "{examples_file}" with train_examples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with train_examples found. Loading it...")
            with open(examples_file, "rb") as f:
                self.train_examples_history = Unpickler(f).load()
            log.info(f"Loading done!")

            # examples based on the model were already collected (loaded)
            self.skip_first_self_play = True
