from __future__ import print_function


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

import pdb


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()

        in_channels = 256
        hidden_dim1 = 64
        hidden_dim2 = 32

        self._conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=num_hiddens, kernel_size=3, stride=1, padding=1)

        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

        self._conv_2 = nn.Conv2d(
            in_channels=num_hiddens, out_channels=num_hiddens // 2, kernel_size=3, stride=2, padding=0
        )

        self._conv_3 = nn.Conv2d(in_channels=num_hiddens // 2, out_channels=1, kernel_size=3, stride=2, padding=0)

        self.fc = nn.Linear(in_channels, hidden_dim1)

        self.rotation_decode = nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 4),
        )
        self.point_decode = nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 3),
            nn.Sigmoid(),
        )
        self.force_decode = nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 3),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        batch_size = inputs.shape[0]

        x = self.fc(inputs)
        x = F.relu(x)

        """
        x = self._conv_1(inputs)
        x = self._residual_stack(x)
        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        x = F.relu(x)

        x = x.view(batch_size, -1)
        """
        rot = self.rotation_decode(x)
        rot = F.normalize(rot, dim=1)

        pos = self.point_decode(x)
        force = self.force_decode(x)
        return rot, pos, force


class TestDecoder(nn.Module):
    """
    p(I|z, O)
    Args:
        latent_dim (int): latent z dimension
        visual_encode_len: observation encoding
        I: interaction parameters=(rotations, force, point) {4+3+3}
    Return:
        mean: [batch_size, class_num, latent_dim]
        logvar: [batch_size, class_num, latent_dim]
    """

    def __init__(self, latent_dim=32, visual_encode_len=128, hidden_dim1=64, hidden_dim2=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.tsdf_vol_bnds = np.array([[-1.5, 0.5], [-1.0, 1.0], [0, 2]])

        self.actvn = F.relu
        self.fc_1 = nn.Linear(visual_encode_len + latent_dim, hidden_dim1)
        # self.fc_2 = nn.Linear(hidden_dim1, hidden_dim1)

        self.rotation_decode = nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 4),
        )
        self.point_decode = nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 3),
            nn.Softmax(),
        )
        self.force_decode = nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 3),
            nn.Softmax(),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.actvn(self.fc_1(x))
        # x = self.actvn(self.fc_2(x))

        rotation = self.rotation_decode(x)
        rotation = F.normalize(rotation, dim=1)

        point = self.point_decode(x)
        force = self.force_decode(x)

        return rotation, point, force
