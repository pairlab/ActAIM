import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from affordance.iafvae.block import ResidualStack, ResnetBlockFC


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

        return rotation, point, force


class LocalDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, n_blocks=2):
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
