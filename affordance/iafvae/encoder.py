import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from affordance.iafvae.block import ResidualStack, ResnetBlockFC


class DepthEncoder(nn.Module):
    """
    Args:
        c_dim (int): output dimension
    """

    def __init__(self, c_dim=128):
        super().__init__()
        self.actvn = F.relu

        self.conv_in = nn.Conv2d(4, 32, 3, padding=1)
        self.conv_in_bn = nn.BatchNorm2d(32)

        self.conv_0 = nn.Conv2d(32, 64, 3, padding=1, stride=3)
        self.conv_0_bn = nn.BatchNorm2d(64)
        self.conv_1 = nn.Conv2d(64, 128, 3, padding=1, stride=3)
        self.conv_1_bn = nn.BatchNorm2d(128)

        self.conv_2 = nn.Conv2d(128, 256, 3, padding=1, stride=3)
        self.conv_2_bn = nn.BatchNorm2d(256)

        self.conv_3 = nn.Conv2d(256, 256, 3, padding=1, stride=3)

        num_residual_hiddens = 32
        num_residual_layers = 2

        self._residual_stack = ResidualStack(
            in_channels=256,
            num_hiddens=256,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

        self.fc = nn.Linear(256 * 4 * 4, c_dim)

    def forward(self, x):
        batch_size = x.size(0)
        if len(x.shape) < 4:
            x = x.unsqueeze(1)

        net = self.conv_in(x)
        net = self.conv_0(self.actvn(self.conv_in_bn(net)))
        net = self.conv_1(self.actvn(self.conv_0_bn(net)))
        net = self.conv_2(self.actvn(self.conv_1_bn(net)))
        net = self.conv_3(self.actvn(self.conv_2_bn(net)))

        net = self._residual_stack(net)

        hidden = net.view(batch_size, 256 * 4 * 4)
        c = torch.sigmoid(self.fc(self.actvn(hidden)))

        return c


class LatentEncode(nn.Module):
    """
    p(z|I, O)
    Args:
        x_dim (int): interaction parameter dimension 3+4+3
        latent_dim (int): latent z dimension
        visual_encode_len: observation encoding
    """

    def __init__(self, latent_dim=64, visual_encode_len=128, x_dim=14, hidden_dim=64):
        super().__init__()
        self.actvn = F.relu
        self.fc_1 = nn.Linear(visual_encode_len + x_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.actvn(self.fc_1(x))
        x = self.actvn(self.fc_2(x))
        x = self.actvn(self.fc_3(x))

        return x
