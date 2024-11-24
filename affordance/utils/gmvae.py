"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------
Gaussian Mixture Variational Autoencoder Networks
"""
import pdb

import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F
import numpy as np
from affordance.utils.layers import *


# Inference Network
class InferenceNet(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super(InferenceNet, self).__init__()

        # q(y|x)
        self.inference_qyx = torch.nn.ModuleList([
            nn.Linear(x_dim, 64),
            nn.ReLU(),
            GumbelSoftmax(64, y_dim)
        ])

        # q(z|y,x)
        self.inference_qzyx = torch.nn.ModuleList([
            nn.Linear(x_dim + y_dim, 64),
            nn.ReLU(),
            Gaussian(64, z_dim)
        ])

    # q(y|x)
    def qyx(self, x, temperature, hard):
        num_layers = len(self.inference_qyx)
        for i, layer in enumerate(self.inference_qyx):
            if i == num_layers - 1:
                # last layer is gumbel softmax
                x = layer(x, temperature, hard)
            else:
                x = layer(x)
        return x

    # q(z|x,y)
    def qzxy(self, x, y):
        concat = torch.cat((x, y), dim=1)
        for layer in self.inference_qzyx:
            concat = layer(concat)
        return concat

    def forward(self, x, temperature=1.0, hard=0):
        # x = Flatten(x)

        # q(y|x)
        logits, prob, y = self.qyx(x, temperature, hard)

        # q(z|x,y)
        mu, var, z = self.qzxy(x, y)

        output = {'mean': mu, 'var': var, 'gaussian': z,
                  'logits': logits, 'prob_cat': prob, 'categorical': y}
        return output


# Generative Network
class GenerativeNet(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super(GenerativeNet, self).__init__()

        output_force_dim = 3
        output_rot_dim = 4

        hidden_dim = 64

        # p(z|y)
        self.y_mu = nn.Linear(y_dim, z_dim)
        self.y_var = nn.Linear(y_dim, z_dim)

        # p(x|z)
        self.generative_pxz = torch.nn.ModuleList([
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Sigmoid()
        ])

        self.fc_r = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_rot_dim)
        )

        self.fc_f = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_force_dim)
        )
        self.sigmoid = nn.Sigmoid()


    # p(z|y)
    def pzy(self, y):
        y_mu = self.y_mu(y)
        y_var = F.softplus(self.y_var(y))
        return y_mu, y_var

    # p(x|z)
    def pxz(self, z):
        for layer in self.generative_pxz:
            z = layer(z)
        return z

    def forward(self, z, y):
        # p(z|y)
        y_mu, y_var = self.pzy(y)

        # p(x|z)
        x_f = self.pxz(z)

        force = self.fc_f(x_f)
        force = self.sigmoid(force)
        rot = self.fc_r(x_f)
        rot = F.normalize(rot, dim=1)

        output = {'y_mean': y_mu, 'y_var': y_var, 'rot':rot, 'force':force}
        return output


# Generative Network
class ConditionalGenerativeNet(nn.Module):
    def __init__(self, x_dim, condition_dim, z_dim, y_dim):
        super(ConditionalGenerativeNet, self).__init__()

        hidden_dim = 640

        # p(z|y)
        self.y_mu = nn.Linear(y_dim, z_dim)
        self.y_var = nn.Linear(y_dim, z_dim)

        # p(skill|z, init_obs)
        self.generative_pxz = torch.nn.ModuleList([
            nn.Linear(z_dim + condition_dim, hidden_dim),
            nn.Sigmoid()
        ])

        self.recon_model = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                         nn.ReLU(0.1),
                                         nn.Linear(hidden_dim, x_dim), nn.Sigmoid()
                                         )

    # p(z|y)
    def pzy(self, y):
        y_mu = self.y_mu(y)
        y_var = F.softplus(self.y_var(y))
        return y_mu, y_var

    # p(x|z)
    def pxz(self, z):
        for layer in self.generative_pxz:
            z = layer(z)
        return z

    def forward(self, condition, z, y):
        # p(z|y)
        y_mu, y_var = self.pzy(y)
        z_condition = torch.cat((z, condition), dim=-1)

        # p(x|z)
        skill = self.pxz(z_condition)
        x_rec = self.recon_model(skill)

        output = {'y_mean': y_mu, 'y_var': y_var, 'x_rec': x_rec, 'skill': skill}
        return output



# GMVAE Network
class GMVAENet(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super(GMVAENet, self).__init__()

        self.inference = InferenceNet(x_dim, z_dim, y_dim)
        self.generative = GenerativeNet(x_dim, z_dim, y_dim)

        # weight initialization
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, temperature=1.0, hard=0):
        x = x.view(x.size(0), -1)
        out_inf = self.inference.forward(x, temperature, hard)
        z, y = out_inf['gaussian'], out_inf['categorical']
        out_gen = self.generative.forward(z, y)

        # merge output
        output = out_inf
        for key, value in out_gen.items():
            output[key] = value
        return output



# GMVAE Network
class CGMVAENet(nn.Module):
    def __init__(self, x_dim, condition_dim, z_dim, y_dim):
        super(CGMVAENet, self).__init__()

        self.inference_net = InferenceNet(x_dim, z_dim, y_dim)
        self.generative = ConditionalGenerativeNet(x_dim, condition_dim, z_dim, y_dim)

        # weight initialization
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, condition, temperature=1.0, hard=0):
        x = x.view(x.size(0), -1)
        out_inf = self.inference_net.forward(x, temperature, hard)
        z, y = out_inf['gaussian'], out_inf['categorical']
        out_gen = self.generative.forward(condition, z, y)

        # merge output
        output = out_inf
        for key, value in out_gen.items():
            output[key] = value
        return output

    def inference(self, condition, noise, category, num_classes=6):
        # categories for each element
        arr = np.array([])
        arr = np.hstack([arr, np.ones(1) * category])
        indices = arr.astype(int).tolist()

        categorical = F.one_hot(torch.tensor(indices), num_classes).float()

        if self.cuda:
            categorical = categorical.cuda()

        # infer the gaussian distribution according to the category
        mean, var = self.generative.pzy(categorical)

        # gaussian random sample by using the mean and variance
        std = torch.sqrt(var)
        gaussian = mean + noise * std

        z_condition = torch.cat((gaussian, condition), dim=-1)
        skill = self.generative.pxz(z_condition)

        return skill
