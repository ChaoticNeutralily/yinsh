"""Adapted from https://github.com/suragnair/alpha-zero-general/blob/master/main.py"""
import logging

import coloredlogs
from torch.cuda import is_available as cuda_is_available

# import sys
# import os

# # getting the name of the directory
# # where the this file is present.
# current = os.path.dirname(os.path.realpath(__file__))

# # Getting the parent directory name
# # where the current directory is present.
# parent = os.path.dirname(current)

# # adding the parent directory to
# # the sys.path.
# sys.path.append(parent)

from yinsh import YinshGame
from alphazero.coach import Coach
from alphazero.nn_utils import YinshNetFlat as flat_nn
from alphazero.yinsh_nnet import YinshNetWrapper as nn


log = logging.getLogger(__name__)

coloredlogs.install(level="INFO")  # Change this to DEBUG to see more info.


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


net_args = dotdict(
    {
        "lr": 0.001,
        "dropout": 0.3,
        "epochs": 10,
        "batch_size": 256,
        "weight_decay": 0.01,
        "cuda": cuda_is_available(),
        "internal_width": 256,
        "internal_layers": 2,
        "external_width": 128,
        "history_length": 1,
    }
)

args = dotdict(
    {
        "first_iter": 2,
        "num_iters": 1000,
        "num_episodes": 100,  # Number of complete self-play games to simulate during a new iteration.
        "temp_threshold": 15,  #
        "update_threshold": 0.6,  # 0.6 During arena playoff, new neural net will be accepted if threshold or more of games are won.
        "maxlen_of_queue": 200000,  # Number of game examples to train the neural networks.
        "num_MCTS_sims": 25,  # Number of games moves for MCTS to simulate.
        "num_arena_games": 40,  # 40 Number of games to play during arena play to determine if new net will be accepted.
        "cpuct": 1,
        "checkpoint": f"checkpoints/{net_args.history_length}x{net_args.internal_layers}x{net_args.internal_width}x{net_args.external_width}/",
        "load_model": True,
        "load_examples": True,
        "load_folder_file": (
            f"checkpoints/{net_args.history_length}x{net_args.internal_layers}x{net_args.internal_width}x{net_args.external_width}/",
            "checkpoint_1.pth.tar",
        ),
        "num_iters_for_train_examples_history": 20,
    }
)


def main():
    log.info("Loading %s...", nn.__name__)
    nnet = nn(flat_nn, net_args)

    if args.load_model:
        log.info(
            'Loading checkpoint "%s/%s"...',
            args.load_folder_file[0],
            args.load_folder_file[1],
        )
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning("Not loading a checkpoint!")

    log.info("Loading the Coach...")
    c = Coach(YinshGame, nnet, args)

    if args.load_examples:
        log.info("Loading 'train_examples' from file...")
        c.load_train_examples()
    c.skip_first_self_play = False
    log.info("Starting the learning process ")
    c.learn()


if __name__ == "__main__":
    main()
