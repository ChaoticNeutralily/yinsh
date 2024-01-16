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

EPS = 1e-10


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


# args = dotdict(
#     {
#         "lr": 0.001,
#         "dropout": 0.3,
#         "epochs": 10,
#         "batch_size": 64,
#         "weight_decay": 0.01,
#         "gpu": "cuda",
#         "internal_layers": 2,
#         "external_width": 256,
#         "history_length": 8,
#     }
# )


class YinshNetWrapper:
    """Adapted from https://github.com/suragnair/alpha-zero-general/blob/master/othello/pytorch/NNet.py

    A wrapper class to train and predict with neural nets that convert yinsh gamestates into (policy, value)
    """

    def __init__(self, net, args):
        self.nnet = net(args)
        self.args = args
        if args.gpu:
            self.gpu_device = torch.device(args.gpu)
            self.nnet.to(self.gpu_device)

        self.value_loss = nn.MSELoss()
        self.history_length = args.history_length
        self.relu = nn.ReLU()

    def policy_loss(self, policy, target):
        policy = torch.log(policy)
        return -torch.mean(torch.flatten(policy) * torch.flatten(target))

    def valid_policy_loss(self, policy, mask):  # uniform_valid_policy):
        # return square of invalid entries
        return torch.mean(torch.flatten((policy * mask) ** 2))
        # + torch.mean(
        #     torch.flatten(self.relu(EPS - policy[uniform_valid_policy > 0]) ** 2)
        # )

    def train(self, examples):
        """
        examples: list of examples, each example is of form
        (state, history, policy, value)
        (preprocessed_gamestate, policy, value)
        """
        self.optimizer = optim.AdamW(
            self.nnet.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        for epoch in range(self.args.epochs):
            print("EPOCH ::: " + str(epoch + 1))
            self.nnet.train()  # sets training to True; makes nnet use dropout, batchnorm etc.
            policy_losses = AverageMeter()
            value_losses = AverageMeter()
            if self.args.show_dummy_loss:
                uniform_policy_losses = AverageMeter()
                uniform_valid_policy_losses = AverageMeter()
                one_value_losses = AverageMeter()
                zero_value_losses = AverageMeter()
                neg_one_value_losses = AverageMeter()

            batch_count = int(len(examples) / self.args.batch_size)

            t = tqdm(range(batch_count), desc="Training Net")
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=self.args.batch_size)
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
                uniform_valid_policies = torch.zeros(target_policies.shape)
                for i in range(self.args.batch_size):
                    uniform_valid_policies[i, :] = self.to_policy(
                        np.ones((len(game_states[i].valid_moves),)),
                        game_states[i].valid_moves,
                    )  # uniform policy over valid moves
                valid_mask = (uniform_valid_policies == 0).type_as(target_policies)
                # policy_mask = torch.sign(uniform_valid_policies)
                # invalid_policy_mask = 1 - policy_mask
                if self.args.show_dummy_loss:
                    # uniform_valid_policies += EPS
                    uniform_policies = torch.full(
                        target_policies.shape,
                        1 / 85,
                    )

                    dummy_v1 = torch.full(
                        target_values.shape,
                        1.0,
                    )
                    dummy_v0 = torch.full(
                        target_values.shape,
                        0.0,
                    )
                    dummy_vn1 = torch.full(
                        target_values.shape,
                        -1.0,
                    )

                    if self.args.gpu:
                        uniform_policies = uniform_policies.to(self.gpu_device)

                        dummy_v1 = dummy_v1.to(self.gpu_device)
                        dummy_v0 = dummy_v0.to(self.gpu_device)
                        dummy_vn1 = dummy_vn1.to(self.gpu_device)

                # predict w/ gpu setup
                if self.args.gpu:
                    # game_states_tensor.to(self.gpu_device)
                    # target_policies.to(self.gpu_device)
                    # target_values.to(self.gpu_device)
                    (
                        game_states_tensor,
                        target_policies,
                        target_values,
                        uniform_valid_policies,
                        valid_mask,
                    ) = (
                        game_states_tensor.contiguous().to(self.gpu_device),
                        target_policies.contiguous().to(self.gpu_device),
                        target_values.contiguous().to(self.gpu_device),
                        uniform_valid_policies.contiguous().to(self.gpu_device),
                        valid_mask.contiguous().to(self.gpu_device),
                    )

                # compute output
                out_policies, out_values = self.nnet(game_states_tensor)
                l_policy = self.policy_loss(
                    out_policies, target_policies.reshape(out_policies.shape)
                )
                l_valid_policy = self.valid_policy_loss(out_policies, valid_mask)
                l_value = self.value_loss(
                    out_values, target_values.reshape(out_values.shape)
                )
                total_loss = l_policy + l_value + 0.1 * l_valid_policy

                # record loss
                policy_losses.update(l_policy.item(), game_states_tensor.size(0))
                value_losses.update(l_value.item(), game_states_tensor.size(0))

                # dummy losses
                if self.args.show_dummy_loss:
                    with torch.no_grad():
                        l_unif_policy = self.policy_loss(
                            uniform_policies,
                            target_policies.reshape(uniform_policies.shape),
                        )
                        l_unif_v_policy = self.policy_loss(
                            uniform_valid_policies + EPS,
                            target_policies.reshape(uniform_valid_policies.shape),
                        )
                        l_0value = self.value_loss(
                            dummy_v1, target_values.reshape(dummy_v1.shape)
                        )
                        l_1value = self.value_loss(
                            dummy_v0, target_values.reshape(dummy_v0.shape)
                        )
                        l_n1value = self.value_loss(
                            dummy_vn1, target_values.reshape(dummy_vn1.shape)
                        )

                        uniform_policy_losses.update(
                            l_unif_policy.item(), game_states_tensor.size(0)
                        )
                        uniform_valid_policy_losses.update(
                            l_unif_v_policy.item(), game_states_tensor.size(0)
                        )
                        one_value_losses.update(
                            l_0value.item(), game_states_tensor.size(0)
                        )
                        zero_value_losses.update(
                            l_1value.item(), game_states_tensor.size(0)
                        )
                        neg_one_value_losses.update(
                            l_n1value.item(), game_states_tensor.size(0)
                        )

                if self.args.show_dummy_loss:
                    t.set_postfix(
                        Loss_policy=policy_losses,
                        Lopu=uniform_policy_losses,
                        Lopuv=uniform_valid_policy_losses,
                        Loss_value=value_losses,
                        vl1=one_value_losses,
                        vl0=zero_value_losses,
                        vln1=neg_one_value_losses,
                    )
                else:
                    t.set_postfix(
                        Loss_policy=policy_losses,
                        Loss_value=value_losses,
                    )

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
            self.save_checkpoint()

    def to_policy(self, probs, moves):
        return self.nnet.postprocessing.to_policy(probs, moves)

    def predict(self, game_state: GameState, history: Iterable[GameBoard]):
        """
        raw game_state and history of boards
        """
        # timing
        start = time.time()

        # preparing input
        game_state_tensor = self.nnet.preprocess(game_state, history)
        if self.args.gpu:
            game_state_tensor = game_state_tensor.contiguous().to(self.gpu_device)
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
        map_location = self.gpu_device if self.args.gpu else "cpu"
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint["state_dict"])
        # self.optimizer.load_state_dict(checkpoint["optim_dict"])
