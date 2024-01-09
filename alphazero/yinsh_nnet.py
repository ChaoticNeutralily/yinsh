import os
import sys
import time

import numpy as np
from tqdm import tqdm
from typing import Iterable

# sys.path.append("../../")

import torch
import torch.nn as nn
import torch.optim as optim

from yinsh import GameState, GameBoard
from alphazero.nn_utils import YinshNetFlat


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f"{self.avg:.2e}"

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


args = dotdict(
    {
        "lr": 0.001,
        "dropout": 0.3,
        "epochs": 10,
        "batch_size": 64,
        "weight_decay": 0.01,
        "cuda": torch.cuda.is_available(),
        "internal_layers": 2,
        "external_width": 256,
        "history_length": 8,
    }
)


class YinshNetWrapper:
    """Adapted from https://github.com/suragnair/alpha-zero-general/blob/master/othello/pytorch/NNet.py

    A wrapper class to train and predict with neural nets that convert yinsh gamestates into (policy, value)
    """

    def __init__(self, net, args):
        self.nnet = net(args)
        self.args = args
        if args.cuda:
            self.nnet.cuda()
        self.value_loss = nn.MSELoss()
        self.log_loss = nn.NLLLoss()
        self.history_length = args.history_length

    def policy_loss(self, policy, target):
        policy = torch.log(policy)
        return -torch.mean(torch.flatten(policy) * torch.flatten(target))

    def train(self, examples):
        """
        examples: list of examples, each example is of form
        (state, history, policy, value)
        (preprocessed_gamestate, policy, value)
        """
        self.optimizer = optim.AdamW(
            self.nnet.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        for epoch in range(args.epochs):
            print("EPOCH ::: " + str(epoch + 1))
            self.nnet.train()  # sets training to True; makes nnet use dropout, batchnorm etc.
            policy_losses = AverageMeter()
            value_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc="Training Net")
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                game_states, histories, policies, values = list(
                    zip(*[examples[i] for i in sample_ids])
                )
                game_states_tensor = torch.FloatTensor(
                    np.array(
                        [
                            self.nnet.preprocess(g, h)
                            for g, h in zip(game_states, histories)
                        ]
                    ).astype(np.float64)
                )
                # print(f"DEBUG: policy length: [len(p) for p in policies]")
                target_policies = torch.FloatTensor(np.array(policies))
                target_values = torch.FloatTensor(np.array(values).astype(np.float64))

                # predict w/ cuda setup
                if args.cuda:
                    game_states_tensor, target_policies, target_values = (
                        game_states_tensor.contiguous().cuda(),
                        target_policies.contiguous().cuda(),
                        target_values.contiguous().cuda(),
                    )

                # compute output
                out_policies, out_values = self.nnet(game_states_tensor)
                l_policy = self.policy_loss(
                    out_policies, target_policies.reshape(out_policies.shape)
                )
                l_value = self.value_loss(
                    out_values, target_values.reshape(out_values.shape)
                )
                total_loss = l_policy + l_value

                # record loss
                policy_losses.update(l_policy.item(), game_states_tensor.size(0))
                value_losses.update(l_value.item(), game_states_tensor.size(0))
                t.set_postfix(Loss_policy=policy_losses, Loss_value=value_losses)
                # For comparison might help to show all 0s, or all 1s policy, and likewise all 0s, all 1s, all -1s value. Then can see if we're doing any better than bad baseline

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

    def to_policy(self, probs, moves):
        return self.nnet.postprocessing.to_policy(probs, moves)

    def predict(self, game_state: GameState, history: Iterable[GameBoard]):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        game_state_tensor = self.nnet.preprocess(game_state, history)
        if args.cuda:
            game_state = game_state.contiguous().cuda()
        self.nnet.eval()  # no dropout or batchnorm
        with torch.no_grad():
            policy, value = self.nnet(game_state_tensor)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return policy.data.cpu().numpy()[0], value.data.cpu().numpy()[0]

    def save_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(
                "Checkpoint Directory does not exist! Making directory {}".format(
                    folder
                )
            )
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save(
            {
                "state_dict": self.nnet.state_dict(),
                # "optim_dict": self.optimizer.state_dict(),
            },
            filepath,
        )

    def load_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise Exception(f"No model in path {filepath}")
        map_location = None if args.cuda else "cpu"
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint["state_dict"])
        # self.optimizer.load_state_dict(checkpoint["optim_dict"])
