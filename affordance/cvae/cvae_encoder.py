import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from affordance.cvae.block import ResidualStack, ResnetBlockFC
import pdb


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)

    return mu + eps * std


class DepthEncoder(nn.Module):
    """
    Args:
        c_dim (int): output dimension
    """

    def __init__(self, c_dim=128, in_channel=4):
        super().__init__()
        self.actvn = F.relu

        self.conv_in = nn.Conv2d(in_channel, 32, 3, padding=1)
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


class CatEncoder(nn.Module):
    """
    p(m|O) ~ Cat(\pi)
    Categorical distribution
    Args:
        c_dim (int): class number 8
        visual_encode_len: observation encoding
    """

    def __init__(self, c_dim=8, visual_encode_len=128):
        super().__init__()
        self.actvn = F.softmax
        self.fc = nn.Linear(visual_encode_len, c_dim)

    def forward(self, x):
        return self.actvn(self.fc(x))


class ModeEncode(nn.Module):
    """
    p(m|I, O) ~ Cat(\pi)
    Categorical distribution
    Args:
        c_dim (int): class number 8
        visual_encode_len: observation encoding
        x_dim (int): interaction parameter dimension 3+4+3
    """

    def __init__(self, c_dim=8, visual_encode_len=128, x_dim=14):
        super().__init__()
        self.actvn = F.softmax
        self.fc = nn.Linear(visual_encode_len + x_dim, c_dim)

    def forward(self, x):
        return self.actvn(self.fc(x))


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

        self.linear_mean = nn.Linear(hidden_dim, latent_dim)
        self.linear_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = self.actvn(self.fc_1(x))
        x = self.actvn(self.fc_2(x))
        x = self.actvn(self.fc_3(x))

        mean = self.linear_mean(x)
        logvar = self.linear_logvar(x)

        return mean, logvar


class ConditionEncode(nn.Module):
    def __init__(self, latent_dim=64, visual_encode_len=128):
        super().__init__()
        self.actvn = F.relu
        self.fc_1 = nn.Linear(visual_encode_len + latent_dim, visual_encode_len)
        self.fc_2 = nn.Linear(visual_encode_len, latent_dim)


    def forward(self, x, c):
        inputs = torch.cat([x, c], 1)

        net = self.actvn(self.fc_1(inputs))
        net = torch.sigmoid(self.fc_2(net))

        return net


class FinetuneEncode(nn.Module):
    def __init__(self, latent_dim=64, visual_encode_len=128):
        super().__init__()
        # self.noise_dim = noise_dim
        self.actvn = F.relu
        self.fc_1 = nn.Linear(visual_encode_len, visual_encode_len)
        self.fc_2 = nn.Linear(visual_encode_len, latent_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        # noise = torch.empty((batch_size, self.noise_dim)).normal_(mean=0, std=1).cuda()
        # inputs = torch.cat([x, noise], 1)

        net = self.actvn(self.fc_1(x))
        # net = torch.sigmoid(self.fc_2(net))
        net = self.fc_2(x)

        return net


class TaskLatentEncode(nn.Module):
    """
    p(z|I, O)
    Args:
        x_dim (int): interaction parameter dimension 3+4+3
        latent_dim (int): latent z dimension
        visual_encode_len: observation encoding
    """

    def __init__(self, latent_dim=64, visual_encode_len=128, hidden_dim=64):
        super().__init__()
        self.actvn = F.relu
        self.fc_1 = nn.Linear(visual_encode_len * 2, visual_encode_len)
        self.fc_2 = nn.Linear(visual_encode_len, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, hidden_dim)

        self.linear_mean = nn.Linear(hidden_dim, latent_dim)
        self.linear_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = self.actvn(self.fc_1(x))
        x = self.actvn(self.fc_2(x))
        x = self.actvn(self.fc_3(x))

        mean = self.linear_mean(x)
        logvar = self.linear_logvar(x)

        return mean, logvar



class ModeLatentEncode(nn.Module):
    """
    p(z|m, O)
    Args:
        c_dim (int): class number 8
        latent_dim (int): latent z dimension
        visual_encode_len: observation encoding

    Return:
        mean: [batch_size, class_num, latent_dim]
        logvar: [batch_size, class_num, latent_dim]
    """

    def __init__(self, latent_dim=64, c_dim=8, visual_encode_len=128, hidden_dim=64):
        super().__init__()
        self.c_dim = c_dim
        self.latent_dim = latent_dim

        self.actvn = F.relu
        self.fc_1 = nn.Linear(visual_encode_len, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)

        self.linear_mean = nn.Linear(hidden_dim, c_dim * latent_dim)
        self.linear_logvar = nn.Linear(hidden_dim, c_dim * latent_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.actvn(self.fc_1(x))
        x = self.actvn(self.fc_2(x))

        mean = self.linear_mean(x)
        logvar = self.linear_logvar(x)

        return mean.reshape([batch_size, self.c_dim, self.latent_dim]), logvar.reshape(
            [batch_size, self.c_dim, self.latent_dim]
        )


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

    def __init__(self, latent_dim=64, visual_encode_len=128, hidden_dim1=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.tsdf_vol_bnds = np.array([[-1.5, 0.5], [-1.0, 1.0], [0, 2]])

        self.actvn = F.relu
        self.fc_1 = nn.Linear(visual_encode_len + latent_dim, hidden_dim1)
        self.fc_2 = nn.Linear(hidden_dim1, hidden_dim1)
        self.fc_3 = nn.Linear(hidden_dim1, hidden_dim1)

        """
        self.rotation_decode = nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 4),
        )
        self.point_decode = nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 3),
            # nn.Sigmoid(),
        )
        self.force_decode = nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 3),
            # nn.Sigmoid(),
        )
        """

        self.rotation_decode = LocalDecoder(hidden_dim1, 4)
        self.point_decode = LocalDecoder(hidden_dim1, 3)
        self.force_decode = LocalDecoder(hidden_dim1, 3)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.actvn(self.fc_1(x))
        x = self.actvn(self.fc_2(x))
        x = self.actvn(self.fc_3(x))

        rotation = self.rotation_decode(x)
        rotation = F.normalize(rotation, dim=1)

        point = self.point_decode(x)
        force = self.force_decode(x)

        # point = torch.sigmoid(point)

        return rotation, point, force


class LocalDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, n_blocks=4):
        super().__init__()

        leaky = True
        self.n_blocks = n_blocks

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.fc_1 = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_dim) for i in range(self.n_blocks)])

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.actvn(self.fc_1(x))

        for i in range(self.n_blocks):
            x = self.blocks[i](x)

        x = self.fc(x)

        return x
