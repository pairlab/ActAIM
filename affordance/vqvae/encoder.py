from __future__ import print_function

import pdb

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


from six.moves import xrange


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from affordance.vqvae.block import ResidualStack


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(
            in_channels=in_channels, out_channels=num_hiddens // 2, kernel_size=4, stride=3, padding=1
        )

        self.batch_norm_1 = nn.BatchNorm2d(num_hiddens // 2, affine=False)

        self._conv_2 = nn.Conv2d(
            in_channels=num_hiddens // 2, out_channels=num_hiddens // 4, kernel_size=4, stride=3, padding=1
        )
        self.batch_norm_2 = nn.BatchNorm2d(num_hiddens // 4, affine=False)

        # Add inter_params embedding
        self._conv_3 = nn.Conv2d(
            in_channels=num_hiddens // 4, out_channels=num_hiddens // 4, kernel_size=4, stride=3, padding=1
        )
        self.batch_norm_3 = nn.BatchNorm2d(num_hiddens // 4, affine=False)

        self._residual_stack = ResidualStack(
            in_channels=num_hiddens // 4,
            num_hiddens=num_hiddens // 4,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

        self.fc_depth = nn.Linear(2304, num_hiddens)
        self.fc_1 = nn.Linear(num_hiddens, num_hiddens * 4)

    def forward(self, inputs, inter_param):
        batch_size = inputs.shape[0]

        x = self._conv_1(inputs)
        x = self.batch_norm_1(x)
        x = F.relu(x)
        # torch.Size([32, 64, 160, 160])

        x = self._conv_2(x)
        x = self.batch_norm_2(x)
        x = F.relu(x)
        # torch.Size([32, 128, 80, 80])

        x = self._conv_3(x)
        x = self.batch_norm_3(x)
        x = F.relu(x)

        x = self._residual_stack(x)
        x = x.view(batch_size, -1)

        x = self.fc_depth(x)
        x = F.relu(x)
        # x = torch.cat((x, inter_param), dim=1)
        x = inter_param
        x = self.fc_1(x)
        x = F.relu(x)

        return x


class InterEncoder(nn.Module):
    def __init__(self, visual_encode_len, hidden_dim):
        super(InterEncoder, self).__init__()

        self.fc_1 = nn.Linear(visual_encode_len, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)

        self._conv_1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        x = self.fc_1(inputs)
        x = F.relu(x)

        x = self.fc_2(x)
        x = F.relu(x)
        """
        x = x.unsqueeze(dim=1)
        x = x.view(-1, 1, 8, 8)
        x = self._conv_1(x)
        x = torch.sigmoid(x)
        x = torch.repeat_interleave(x, 10, dim=2)
        x = torch.repeat_interleave(x, 10, dim=-1)
        """

        return x
