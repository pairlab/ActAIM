from __future__ import print_function


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import argparse

from six.moves import xrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from new_scripts.model.task.multi_view_encoder import MultiViewEncoder
from new_scripts.model.task.multi_view_encoder import ResnetBlockFC
from new_scripts.model.task.networks.Networks import GMVAENet
from new_scripts.model.task.networks.Losses import LossFunctions
import pdb

def create_parser():
    parser = argparse.ArgumentParser()
    ## Architecture
    parser.add_argument('--num_classes', type=int, default=8,
                    help='number of classes (default: 8)')
    parser.add_argument('--gaussian_size', default=64, type=int,
                    help='gaussian size (default: 64)')
    parser.add_argument('--input_size', default=640, type=int,
                    help='input size (default: 640)')

    ## Gumbel parameters
    parser.add_argument('--init_temp', default=1.0, type=float,
                    help='Initial temperature used in gumbel-softmax (recommended 0.5-1.0, default:1.0)')
    parser.add_argument('--decay_temp', default=1, type=int,
                    help='Set 1 to decay gumbel temperature at every epoch (default: 1)')
    parser.add_argument('--hard_gumbel', default=0, type=int,
                    help='Set 1 to use the hard version of gumbel-softmax (default: 1)')
    parser.add_argument('--min_temp', default=0.5, type=float,
                    help='Minimum temperature of gumbel-softmax after annealing (default: 0.5)' )
    parser.add_argument('--decay_temp_rate', default=0.013862944, type=float,
                    help='Temperature decay rate at every epoch (default: 0.013862944)')

    ## Loss function parameters
    parser.add_argument('--w_gauss', default=3, type=float,
                    help='weight of gaussian loss (default: 1)')
    parser.add_argument('--w_categ', default=3, type=float,
                    help='weight of categorical loss (default: 1)')
    parser.add_argument('--w_rec', default=2, type=float,
                    help='weight of reconstruction loss (default: 1)')
    parser.add_argument('--rec_type', type=str, choices=['bce', 'mse'],
                    default='mse', help='desired reconstruction loss function (default: bce)')
    return parser




class CGmvaeTransformer(nn.Module):
    def __init__(
        self,
        latent_dim=512,
    ):
        super(CGmvaeTransformer, self).__init__()

        # loss params
        self.w_cat = 3
        self.w_gauss = 3
        self.w_rec = 2
        self.rec_type = 'mse' # ['bce', 'mse']

        # Architecture params
        self.num_classes = 8
        self.gaussian_size = 64
        self.input_size = 640

        # gumbel
        self.init_temp = 1.0
        self.decay_temp = 1
        self.hard_gumbel = 0
        self.min_temp = 0.5
        self.decay_temp_rate = 0.013862944
        self.gumbel_temp = self.init_temp

        self.network = GMVAENet(self.input_size, self.gaussian_size, self.num_classes)
        self.losses = LossFunctions()

        self.multi_view_encode = MultiViewEncoder(latent_dim)

    def unlabeled_loss(self, data, out_net):
        """Method defining the loss functions derived from the variational lower bound
        Args:
            data: (array) corresponding array containing the input data
            out_net: (dict) contains the graph operations or nodes of the network output

        Returns:
            loss_dic: (dict) contains the values of each loss function and predictions
        """
        # obtain network variables
        z, data_recon = out_net['gaussian'], out_net['x_rec']
        logits, prob_cat = out_net['logits'], out_net['prob_cat']
        y_mu, y_var = out_net['y_mean'], out_net['y_var']
        mu, var = out_net['mean'], out_net['var']

        # contrastive loss given label
        # loss_contrast = self.losses.contrastive_loss(x_learned, labels)

        # reconstruction loss
        loss_rec = self.losses.reconstruction_loss(data, data_recon, self.rec_type)

        # gaussian loss
        loss_gauss = self.losses.gaussian_loss(z, mu, var, y_mu, y_var)

        # categorical loss
        loss_cat = -self.losses.entropy(logits, prob_cat) - np.log(0.1)

        # total loss
        loss_total = self.w_rec * loss_rec + self.w_gauss * loss_gauss + self.w_cat * loss_cat

        # obtain predictions
        _, predicted_labels = torch.max(logits, dim=1)

        loss_dic = {'total': loss_total,
                    'predicted_labels': predicted_labels,
                    'reconstruction': self.w_rec * loss_rec,
                    'gaussian': self.w_gauss * loss_gauss,
                    'categorical': self.w_cat * loss_cat,
                    }

        return loss_dic

    def forward(self, curr_obs, final_obs, epoch):
        out_net = self.network(curr_obs, final_obs, self.gumbel_temp, self.hard_gumbel)
        unlab_loss_dic = self.unlabeled_loss(out_net["data"], out_net)
        total_loss = unlab_loss_dic['total']

        # decay gumbel temperature
        if self.decay_temp == 1:
            self.gumbel_temp = np.maximum(self.init_temp * np.exp(-self.decay_temp_rate * epoch), self.min_temp)


        task_embed = out_net["x_rec"]
        task_label = out_net["data"]
        final_x = out_net["final_x"]

        return total_loss.mean(), task_embed, None, task_label, final_x

    def sample(self, curr_obs, num_elements=1):
        cond_x = self.network.vision_encoder(curr_obs)
        # categories for each element
        arr = np.array([])
        for i in range(self.num_classes):
            arr = np.hstack([arr, np.ones(num_elements) * i])
        indices = arr.astype(int).tolist()

        categorical = F.one_hot(torch.tensor(indices), self.num_classes).float()

        if self.cuda:
            categorical = categorical.cuda()

        if cond_x.shape[0] != num_elements:
            cond_x = cond_x.repeat(categorical.shape[0], 1, 1)

        # infer the gaussian distribution according to the category
        mean, var = self.network.generative.pzy(categorical)

        # gaussian random sample by using the mean and variance
        noise = torch.randn_like(var)
        std = torch.sqrt(var)
        gaussian = mean + noise * std

        # generate new samples with the given gaussian
        generated = self.network.generative.pxz_cond(gaussian, cond_x)

        return generated.detach()
