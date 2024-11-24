from __future__ import print_function

import pdb

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


from six.moves import xrange


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_residual_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens, out_channels=num_hiddens, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [Residual(in_channels, num_hiddens, num_residual_hiddens) for _ in range(self._num_residual_layers)]
        )

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    """Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(x.shape[0], *self.shape)


class MaskedLinear(nn.Linear):
    """Masked linear layer"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", self.weight.data.clone())
        H, W = self.weight.size()
        self.mask.fill_(1)
        for line in range(H):
            self.mask[line, line:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class ARBlock(nn.Module):
    def __init__(self, in_size, context_size, out_size, n_layers=2, act=F.elu):
        super().__init__()
        self.act = act
        layers = []

        for i in range(n_layers):
            layers.append(MaskedLinear(in_size if i == 0 else context_size, context_size))
        layers.append(MaskedLinear(context_size, out_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, context):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == 0:
                x += context
            x = self.act(x)

        return x


class IAFBlock(nn.Module):
    def __init__(self, input_size, z_size):
        super().__init__()
        self.input_size = input_size
        self.z_size = z_size
        self.h_size = input_size

        self.up_a = nn.Linear(input_size, z_size * 2 + self.h_size * 2)
        self.up_b = nn.Linear(self.h_size, self.h_size)
        self.down_a = nn.Linear(self.h_size, 4 * z_size + 2 * self.h_size)
        self.down_ar = ARBlock(z_size, self.h_size, 2 * z_size)
        self.down_b = nn.Linear(self.h_size + self.z_size, self.h_size)

    def up(self, input):
        x = self.up_a(F.elu(input))
        self.qz_mean, self.qz_logsd, self.up_context, h = x.split([self.z_size] * 2 + [self.h_size] * 2, 1)

        h = F.elu(h)
        h = self.up_b(h)

        return input + 0.1 * h

    def down(self, input, sample=False):
        x = F.elu(input)
        x = self.down_a(x)

        pz_mean, pz_logsd, rz_mean, rz_logsd, down_context, h_det = x.split([self.z_size] * 4 + [self.h_size] * 2, 1)
        prior = td.Normal(pz_mean, torch.exp(2 * pz_logsd))

        if sample:
            z = prior.rsample()
            kl = kl_obj = torch.zeros(input.size(0)).to(input.device)
        else:
            posterior = td.Normal(rz_mean + self.qz_mean, torch.exp(rz_logsd + self.qz_logsd))

            z = posterior.rsample()
            logqs = posterior.log_prob(z)
            context = self.up_context + down_context

            x = self.down_ar(z, context)
            arw_mean, arw_logsd = (x * 0.1).split([self.z_size] * 2, 1)
            z = (z - arw_mean) / torch.exp(arw_logsd)

            # the log density at the new point is the old one - log determinant of transformation
            logqs += arw_logsd

            logps = prior.log_prob(z)
            kl = logqs - logps

            # # free bits (doing as in the original repo, even if weird)
            # kl_obj = kl.sum(dim=(-2, -1)).mean(dim=0, keepdim=True)
            # kl_obj = kl_obj.clamp(min=self.args.free_bits)
            # kl_obj = kl_obj.expand(kl.size(0), -1)
            # kl_obj = kl_obj.sum(dim=1)

            # sum over all the dimensions, but the batch
            kl = kl.sum(dim=(1,))

        h = torch.cat((z, h_det), 1)
        h = F.elu(h)
        h = self.down_b(h)

        return input + 0.1 * h, kl


class IAFlow(nn.Module):
    def __init__(self, input_size, n_layers, z_size):
        super().__init__()
        self.input_size = input_size
        self.z_size = z_size
        self.h_size = input_size
        self.register_parameter("h", torch.nn.Parameter(torch.zeros(input_size)))
        self.blocks = nn.ModuleList([IAFBlock(input_size, z_size) for _ in range(n_layers)])

    def forward(self, input):
        x = input
        for block in self.blocks:
            x = block.up(x)

        h = self.h.view(1, -1).expand_as(x)
        kl = 0.0
        for block in self.blocks:
            h, curr_kl = block.down(h, sample=False)
            kl += curr_kl

        return h, kl

    def sample(self, n_samples=4):
        h = self.h.view(1, -1).expand((n_samples, self.h_size))
        for block in self.blocks:
            h, _ = block.down(h, sample=True)

        return h
