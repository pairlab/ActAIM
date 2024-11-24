import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from affordance.cvae.cvae_encoder import *
from affordance.cvae.block import ResidualStack, ResnetBlockFC
from affordance.utils.gmvae import GMVAENet
from affordance.utils.LossFunctions import LossFunctions
from affordance.utils.layers import Gaussian

class PointCritic(nn.Module):
    def __init__(self, output_dim=1, latent_dim=64, feature_dim=96, hidden_dim=64, n_blocks=2):
        super().__init__()

        leaky = True
        self.n_blocks = n_blocks

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        input_dim_concat = latent_dim + feature_dim
        self.sigmoid = nn.Sigmoid()

        self.fc_0 = nn.Linear(input_dim_concat, hidden_dim * 2)
        self.fc_1 = nn.Linear(hidden_dim * 2, hidden_dim)

        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_dim) for i in range(self.n_blocks)])

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, pos_feature):
        x = torch.cat((pos_feature, z), dim=-1)
        x = self.actvn(self.fc_0(x))
        x = self.actvn(self.fc_1(x))

        for i in range(self.n_blocks):
            x = self.blocks[i](x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class PointCritic2(nn.Module):
    def __init__(self, output_dim=1, feature_dim=96, hidden_dim=64, n_blocks=2):
        super().__init__()

        leaky = True
        self.n_blocks = n_blocks

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        input_dim_concat = feature_dim
        self.sigmoid = nn.Sigmoid()

        self.fc_0 = nn.Linear(input_dim_concat, hidden_dim * 2)
        self.fc_1 = nn.Linear(hidden_dim * 2, hidden_dim)

        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_dim) for i in range(self.n_blocks)])

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, pos_feature):
        x = self.actvn(self.fc_0(pos_feature))
        x = self.actvn(self.fc_1(x))

        for i in range(self.n_blocks):
            x = self.blocks[i](x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class RotPolicy(nn.Module):
    def __init__(self, obs_dim=128, output_dim=4, latent_dim=64, feature_dim=96, hidden_dim=64, n_blocks=2):
        super().__init__()
        self.obs_encoder = DepthEncoder(c_dim=obs_dim)

        leaky = True
        self.n_blocks = n_blocks

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        input_dim_concat = obs_dim + latent_dim + feature_dim

        self.fc_0 = nn.Linear(input_dim_concat, hidden_dim * 2)
        self.fc_1 = nn.Linear(hidden_dim * 2, hidden_dim)

        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_dim) for i in range(self.n_blocks)])

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, obs, pos_feature):
        obs_encode = self.obs_encoder(obs)
        x = torch.cat((obs_encode, pos_feature, z), dim=-1)
        x = self.actvn(self.fc_0(x))
        x = self.actvn(self.fc_1(x))

        for i in range(self.n_blocks):
            x = self.blocks[i](x)

        x = self.fc(x)
        x = F.normalize(x, dim=1)
        return x


class ForcePolicy(nn.Module):
    def __init__(self, obs_dim=128, output_dim=3, latent_dim=64, feature_dim=96, hidden_dim=64, n_blocks=2):
        super().__init__()
        self.obs_encoder = DepthEncoder(c_dim=obs_dim)

        leaky = True
        self.n_blocks = n_blocks

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        input_dim_concat = obs_dim + latent_dim + feature_dim + 4

        self.fc_0 = nn.Linear(input_dim_concat, hidden_dim * 2)
        self.fc_1 = nn.Linear(hidden_dim * 2, hidden_dim)

        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_dim) for i in range(self.n_blocks)])

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, obs, pos_feature, rot):
        obs_encode = self.obs_encoder(obs)
        x = torch.cat((obs_encode, pos_feature, z, rot), dim=-1)
        x = self.actvn(self.fc_0(x))
        x = self.actvn(self.fc_1(x))
        for i in range(self.n_blocks):
            x = self.blocks[i](x)

        x = self.fc(x)
        return x


class StochasticActionPolicy(nn.Module):
    def __init__(self, obs_dim=128, output_force_dim=3, output_rot_dim = 4, latent_dim=64, feature_dim=96, hidden_dim=64, n_blocks=2):
        super().__init__()

        self.log_std_min = -10
        self.log_std_max = 2


        leaky = True
        self.n_blocks = n_blocks

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        input_dim_concat = obs_dim + latent_dim + feature_dim

        self.fc_0 = nn.Linear(input_dim_concat, hidden_dim * 2)
        self.fc_1 = nn.Linear(hidden_dim * 2, hidden_dim)

        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_dim) for i in range(self.n_blocks)])

        self.fc_r = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_rot_dim * 2)
        )

        self.fc_f = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_force_dim * 2)
        )


        self.sigmoid = nn.Sigmoid()

    def forward(self, z, obs, pos_feature, obs_encoder):
        obs_encode = obs_encoder.forward(obs)
        x = torch.cat((obs_encode, pos_feature, z), dim=-1)
        x = self.actvn(self.fc_0(x))
        x = self.actvn(self.fc_1(x))
        for i in range(self.n_blocks):
            x = self.blocks[i](x)

        force_mu, force_log_std = self.fc_f(x).chunk(2, dim=-1)
        rot_mu, rot_log_std = self.fc_r(x).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        force_log_std = torch.tanh(force_log_std)
        force_log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (force_log_std + 1)

        rot_log_std = torch.tanh(rot_log_std)
        rot_log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (rot_log_std + 1)

        force_std = force_log_std.exp()
        noise = torch.randn_like(force_mu)
        force = force_mu + noise * force_std
        force = self.sigmoid(force)

        rot_std = rot_log_std.exp()
        noise = torch.randn_like(rot_mu)
        rot = rot_mu + noise * rot_std
        rot = F.normalize(rot, dim=1)

        return rot, force


class ActionPolicy(nn.Module):
    def __init__(self, obs_dim=128, output_force_dim=3, output_rot_dim = 4, latent_dim=64, feature_dim=96, hidden_dim=64, n_blocks=2):
        super().__init__()

        leaky = True
        self.n_blocks = n_blocks

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        input_dim_concat = obs_dim + latent_dim + feature_dim

        self.fc_0 = nn.Linear(input_dim_concat, hidden_dim * 2)
        self.fc_1 = nn.Linear(hidden_dim * 2, hidden_dim)

        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_dim) for i in range(self.n_blocks)])

        self.fc_r = nn.Linear(hidden_dim, output_rot_dim)
        self.fc_f = nn.Linear(hidden_dim, output_force_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, z, obs, pos_feature, obs_encoder):
        obs_encode = obs_encoder.forward(obs)
        x = torch.cat((obs_encode, pos_feature, z), dim=-1)
        x = self.actvn(self.fc_0(x))
        x = self.actvn(self.fc_1(x))
        for i in range(self.n_blocks):
            x = self.blocks[i](x)

        force = self.fc_f(x)
        force = self.sigmoid(force)
        rot = self.fc_r(x)
        rot = F.normalize(rot, dim=1)
        return rot, force

class ScoreCritic(nn.Module):
    def __init__(self, output_dim=1, latent_dim=64, feature_dim=96, hidden_dim=64, n_blocks=4):
        super().__init__()

        leaky = True
        self.n_blocks = n_blocks

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        input_dim_concat = feature_dim
        self.sigmoid = nn.Sigmoid()
        self.action_projection = nn.Linear(8+3, 8+3)

        self.fc_0 = nn.Linear(latent_dim + input_dim_concat + 8 + 3, hidden_dim * 2)
        self.fc_1 = nn.Linear(hidden_dim * 2, hidden_dim)

        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_dim) for i in range(self.n_blocks)])

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, skill, pos_feature, r, f):
        r = r.view(r.shape[0], -1)
        action = torch.cat((r, f), dim=-1)
        action = self.actvn(self.action_projection(action))
        x = torch.cat((skill, pos_feature, action), dim=-1)

        x = self.actvn(self.fc_0(x))
        x = self.actvn(self.fc_1(x))

        for i in range(self.n_blocks):
            x = self.blocks[i](x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class ScoreCritic2(nn.Module):
    def __init__(self, output_dim=1, feature_dim=96, hidden_dim=64, n_blocks=4):
        super().__init__()

        leaky = True
        self.n_blocks = n_blocks

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        input_dim_concat = feature_dim
        self.sigmoid = nn.Sigmoid()
        self.action_projection = nn.Linear(8+3, 8+3)

        self.fc_0 = nn.Linear(input_dim_concat + 8 + 3, hidden_dim * 2)
        self.fc_1 = nn.Linear(hidden_dim * 2, hidden_dim)

        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_dim) for i in range(self.n_blocks)])

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, pos_feature, r, f):
        r = r.view(r.shape[0], -1)
        action = torch.cat((r, f), dim=-1)
        action = self.actvn(self.action_projection(action))
        x = torch.cat((pos_feature, action), dim=-1)

        x = self.actvn(self.fc_0(x))
        x = self.actvn(self.fc_1(x))

        for i in range(self.n_blocks):
            x = self.blocks[i](x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class VAEActionPolicy(nn.Module):
    def __init__(self, obs_dim=128, output_force_dim=3, output_rot_dim = 4, latent_dim=64, feature_dim=96, hidden_dim=64, n_blocks=2):
        super().__init__()

        leaky = True
        self.n_blocks = n_blocks

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        input_dim_concat = obs_dim + latent_dim + feature_dim

        self.fc_0 = nn.Linear(input_dim_concat, hidden_dim * 2)
        self.fc_1 = nn.Linear(hidden_dim * 2, hidden_dim)

        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_dim) for i in range(self.n_blocks)])
        self.gaussian = Gaussian(hidden_dim, hidden_dim // 2)

        self.fc_r = nn.Linear(hidden_dim // 2, output_rot_dim)
        self.fc_f = nn.Linear(hidden_dim // 2, output_force_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, z, obs, pos_feature, obs_encoder):
        obs_encode = obs_encoder.forward(obs)
        x = torch.cat((obs_encode, pos_feature, z), dim=-1)
        x = self.actvn(self.fc_0(x))
        x = self.actvn(self.fc_1(x))
        for i in range(self.n_blocks):
            x = self.blocks[i](x)

        mu, var, x = self.gaussian.forward(x)

        force = self.fc_f(x)
        force = self.sigmoid(force)
        rot = self.fc_r(x)
        rot = F.normalize(rot, dim=1)
        return rot, force


    def inference(self, z, obs, pos_feature, obs_encoder):
        obs_encode = obs_encoder.forward(obs)
        x = torch.cat((obs_encode, pos_feature, z), dim=-1)
        x = self.actvn(self.fc_0(x))
        x = self.actvn(self.fc_1(x))
        for i in range(self.n_blocks):
            x = self.blocks[i](x)

        mu, var, x = self.gaussian.forward(x)

        force = self.fc_f(x)
        force = self.sigmoid(force)
        rot = self.fc_r(x)
        rot = F.normalize(rot, dim=1)
        return rot, force




class GMVAEActionPolicy(nn.Module):
    def __init__(self, obs_dim=128, feature_dim=96, hidden_dim=64, n_blocks=2):
        super().__init__()

        self.log_std_min = -10
        self.log_std_max = 2


        leaky = True
        self.n_blocks = n_blocks

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        input_dim_concat = obs_dim + feature_dim

        self.fc_0 = nn.Linear(input_dim_concat, hidden_dim * 2)
        self.fc_1 = nn.Linear(hidden_dim * 2, hidden_dim)

        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_dim) for i in range(self.n_blocks)])

        # y_dim = 32, num_classes = 2
        self.gmvae = GMVAENet(hidden_dim, 32, 2)
        self.losses = LossFunctions()

        self.sigmoid = nn.Sigmoid()
        self.w_gauss = 2
        self.w_cat = 2


    def forward(self, obs, pos_feature, obs_encoder):
        obs_encode = obs_encoder.forward(obs)
        x = torch.cat((obs_encode, pos_feature), dim=-1)
        x = self.actvn(self.fc_0(x))
        x = self.actvn(self.fc_1(x))
        for i in range(self.n_blocks):
            x = self.blocks[i](x)

        # input dim = hiddem_dim = 64
        out_net = self.gmvae.forward(x)

        z = out_net['gaussian']
        logits, prob_cat = out_net['logits'], out_net['prob_cat']
        y_mu, y_var = out_net['y_mean'], out_net['y_var']
        mu, var = out_net['mean'], out_net['var']
        rot, force = out_net['rot'], out_net['force']

        # gaussian loss
        loss_gauss = self.losses.gaussian_loss(z, mu, var, y_mu, y_var)

        # categorical loss
        loss_cat = -self.losses.entropy(logits, prob_cat) - np.log(0.1)

        kl_loss = self.w_gauss * loss_gauss + self.w_cat * loss_cat

        return rot, force, kl_loss