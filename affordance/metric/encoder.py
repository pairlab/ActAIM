import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxelEncoder(nn.Module):
    """3D-convolutional encoder network for voxel input.

    Args:
        dim (int): input dimension
        c_dim (int): output dimension
    """

    def __init__(self, dim=3, c_dim=16):
        super().__init__()
        self.actvn = F.relu

        self.conv_in = nn.Conv3d(1, 32, 3, padding=1)

        self.conv_0 = nn.Conv3d(32, 64, 3, padding=1, stride=2)
        self.conv_1 = nn.Conv3d(64, 128, 3, padding=1, stride=2)
        self.conv_2 = nn.Conv3d(128, 256, 3, padding=1, stride=2)
        self.fc = nn.Linear(256 * 5 * 5 * 5, c_dim)

    def forward(self, x):
        batch_size = x.size(0)

        x = x.unsqueeze(1)
        net = self.conv_in(x)
        net = self.conv_0(self.actvn(net))
        net = self.conv_1(self.actvn(net))
        net = self.conv_2(self.actvn(net))

        hidden = net.view(batch_size, 256 * 5 * 5 * 5)
        c = torch.sigmoid(self.fc(self.actvn(hidden)))

        return c


class PixelEncoder(nn.Module):
    """3D-convolutional encoder network for voxel input.

    Args:
        dim (int): input dimension
        c_dim (int): output dimension
    """

    def __init__(self, dim=3, c_dim=16):
        super().__init__()
        self.actvn = F.relu

        self.conv_in = nn.Conv2d(4, 32, 3, padding=1)

        self.conv_0 = nn.Conv2d(32, 64, 3, padding=1, stride=3)
        self.conv_1 = nn.Conv2d(64, 128, 3, padding=1, stride=3)
        self.conv_2 = nn.Conv2d(128, 256, 3, padding=1, stride=3)
        self.conv_3 = nn.Conv2d(256, 256, 3, padding=1, stride=3)
        self.fc = nn.Linear(256 * 4 * 4, c_dim)

    def forward(self, x):
        batch_size = x.size(0)

        net = self.conv_in(x)
        net = self.conv_0(self.actvn(net))
        net = self.conv_1(self.actvn(net))
        net = self.conv_2(self.actvn(net))
        net = self.conv_3(self.actvn(net))
        hidden = net.view(batch_size, 256 * 4 * 4)
        c = torch.sigmoid(self.fc(self.actvn(hidden)))

        return c


class DepthEncoder(nn.Module):
    """3D-convolutional encoder network for voxel input.

    Args:
        dim (int): input dimension
        c_dim (int): output dimension
    """

    def __init__(self, dim=3, c_dim=16):
        super().__init__()
        self.actvn = F.relu

        self.conv_in = nn.Conv2d(1, 32, 3, padding=1)

        self.conv_0 = nn.Conv2d(32, 64, 3, padding=1, stride=3)
        self.conv_1 = nn.Conv2d(64, 128, 3, padding=1, stride=3)
        self.conv_2 = nn.Conv2d(128, 256, 3, padding=1, stride=3)
        self.conv_3 = nn.Conv2d(256, 256, 3, padding=1, stride=3)
        self.fc = nn.Linear(256 * 4 * 4, c_dim)

    def forward(self, x):
        batch_size = x.size(0)

        x = x.unsqueeze(1)
        net = self.conv_in(x)
        net = self.conv_0(self.actvn(net))
        net = self.conv_1(self.actvn(net))
        net = self.conv_2(self.actvn(net))
        net = self.conv_3(self.actvn(net))
        hidden = net.view(batch_size, 256 * 4 * 4)
        c = torch.sigmoid(self.fc(self.actvn(hidden)))

        return c


class AutoEncoder(nn.Module):
    """3D-convolutional encoder network for pixel input.

    Args:
        dim (int): input dimension
        c_dim (int): output dimension
    """

    def __init__(self, c_dim=128):
        super().__init__()
        self.actvn = F.relu

        self.conv_in = nn.Conv2d(1, 32, 3, padding=1)
        self.conv_0 = nn.Conv2d(32, 64, 3, padding=1, stride=3)
        self.conv_1 = nn.Conv2d(64, 128, 3, padding=1, stride=3)
        self.conv_2 = nn.Conv2d(128, 256, 3, padding=1, stride=3)
        self.conv_3 = nn.Conv2d(256, 256, 3, padding=1, stride=3)
        self.fc = nn.Linear(256 * 4 * 4, c_dim)

        self.dec_fc = nn.Linear(c_dim, 256 * 4 * 4)
        self.dec_conv_3 = nn.ConvTranspose2d(256, 256, 3, padding=0, stride=3)
        self.dec_conv_2 = nn.ConvTranspose2d(256, 128, 3, padding=0, stride=3)
        self.dec_conv_1 = nn.ConvTranspose2d(128, 64, 3, padding=0, stride=3)
        self.dec_conv_0 = nn.ConvTranspose2d(64, 32, 3, padding=2, stride=3)
        self.dec_conv_out = nn.ConvTranspose2d(32, 1, 3, padding=1)

    def encode(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        net = self.conv_in(x)
        net = self.conv_0(self.actvn(net))
        net = self.conv_1(self.actvn(net))
        net = self.conv_2(self.actvn(net))
        net = self.conv_3(self.actvn(net))
        hidden = net.view(batch_size, 256 * 4 * 4)
        c = torch.sigmoid(self.fc(self.actvn(hidden)))

        return c

    def decode(self, x):
        import pdb

        batch_size = x.size(0)
        x = self.dec_fc(x)
        x = x.view(batch_size, 256, 4, 4)
        x = self.dec_conv_3(x)
        x = self.dec_conv_2(x)
        x = self.dec_conv_1(x)
        x = self.dec_conv_0(x)
        x = self.dec_conv_out(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        _x = self.decode(z)
        _x = _x.squeeze()
        return _x


class Classifier(nn.Module):
    def __init__(self, c_dim=128):
        super().__init__()
        self.actvn = F.relu

        self.fc_1 = nn.Linear(c_dim * 2, c_dim // 2)
        self.fc_2 = nn.Linear(c_dim // 2, c_dim // 8)
        self.fc_3 = nn.Linear(c_dim // 8, 1)

    def forward(self, x, y):
        batch_size = x.size(0)
        net = torch.cat((x, y), -1)

        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.fc_3(net)

        net = torch.sigmoid(net)
        return net
