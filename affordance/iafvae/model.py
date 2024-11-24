from __future__ import print_function


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from affordance.iafvae.encoder import DepthEncoder, LatentEncode
from affordance.iafvae.decoder import TestDecoder
from affordance.iafvae.block import IAFlow
import pdb


class IAFModel(nn.Module):
    def __init__(self):
        super(IAFModel, self).__init__()
        self.decoder = TestDecoder(latent_dim=64, visual_encode_len=64, hidden_dim1=64)  # p(I|z,o)
        self.obs_encoder = DepthEncoder(c_dim=64)
        self.flow = IAFlow(64, n_layers=3, z_size=32)
        self.latent = LatentEncode(latent_dim=64, visual_encode_len=64, x_dim=14, hidden_dim=64)  # p(z|I, o)

    def forward(self, inter_params, obs):
        obs_encode = self.obs_encoder(obs)
        obs_inter = torch.cat((obs_encode, inter_params), dim=-1)
        x = self.latent(obs_inter)
        h, kl = self.flow(x)
        obs_h = torch.cat((obs_encode, h), dim=-1)
        (recon_rotations, recon_point, recon_force) = self.decoder(obs_h)

        return kl, (recon_rotations, recon_point, recon_force)

    def sample(self, n_samples=4):
        h = self.flow.sample(n_samples)
        output = self.decoder(h)
        return output
