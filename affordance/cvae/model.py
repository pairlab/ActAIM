from __future__ import print_function


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from affordance.cvae.block import ResidualStack, ResnetBlockFC
from affordance.utils.gmvae import CGMVAENet
from affordance.utils.LossFunctions import LossFunctions
from affordance.cvae.cvae_encoder import *
import pdb


class CVAEModel(nn.Module):
    def __init__(self):
        super(CVAEModel, self).__init__()
        self.decoder = TestDecoder(latent_dim=256, visual_encode_len=128, hidden_dim1=128)  # p(I|z,o)
        self.obs_encoder = DepthEncoder(c_dim=128)
        self.posterior_z = LatentEncode(latent_dim=256, visual_encode_len=128, x_dim=14, hidden_dim=128)  # p(z|I, o)

    def forward(self, inter_params, obs, eval=False):
        batch_size = obs.shape[0]
        obs_encode = self.obs_encoder(obs)
        obs_inter = torch.cat((obs_encode, inter_params), dim=-1)
        z_mean, z_log_var = self.posterior_z(obs_inter)

        z = reparameterize(z_mean, z_log_var)

        if eval:
            z = torch.empty((batch_size, 256)).normal_(mean=0, std=1).cuda()
        obs_z = torch.cat((obs_encode, z), dim=-1)
        recon_rotations, recon_point, recon_force = self.decoder(obs_z)

        kl_loss_z = -0.5 * (1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))
        kl_loss_z = torch.mean(kl_loss_z)

        return kl_loss_z, (z_mean, z_log_var), (recon_rotations, recon_point, recon_force), z

class TaskCVAEModelBaseline(nn.Module):
    def __init__(self):
        super(TaskCVAEModelBaseline, self).__init__()
        self.decoder = TestDecoder(latent_dim=256, visual_encode_len=128, hidden_dim1=128)  # p(I|z,o)
        self.obs_encoder = DepthEncoder(c_dim=128)
        self.next_obs_encode = DepthEncoder(c_dim=128, in_channel=1)
        self.posterior_z = LatentEncode(latent_dim=256, visual_encode_len=128, x_dim=128, hidden_dim=128)  # p(z|I, o)

    def forward(self, inter_params, obs, next_obs, eval=False):
        batch_size = obs.shape[0]
        obs_encode = self.obs_encoder(obs)
        if len(next_obs.shape) == 3:
            next_obs = torch.unsqueeze(next_obs, 1)
        next_obs_encode = self.next_obs_encode(next_obs)
        obs_inter = torch.cat((obs_encode, next_obs_encode), dim=-1)

        z_mean, z_log_var = self.posterior_z(obs_inter)

        z = reparameterize(z_mean, z_log_var)

        if eval:
            z = torch.empty((batch_size, 256)).normal_(mean=0, std=1).cuda()
        obs_z = torch.cat((obs_encode, z), dim=-1)
        recon_rotations, recon_point, recon_force = self.decoder(obs_z)

        kl_loss_z = -0.5 * (1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))
        kl_loss_z = torch.mean(kl_loss_z)

        return kl_loss_z, (z_mean, z_log_var), (recon_rotations, recon_point, recon_force), z


class TaskCVAEModel_old(nn.Module):
    def __init__(self):
        super(TaskCVAEModel, self).__init__()
        self.obs_encoder = DepthEncoder(c_dim=128)
        self.next_obs_encode = DepthEncoder(c_dim=128, in_channel=1)
        self.posterior_z = LatentEncode(latent_dim=64, visual_encode_len=128, x_dim=128, hidden_dim=128)  # p(z|I, o)

    def forward(self, inter_params, obs, next_obs, eval=False):
        batch_size = obs.shape[0]
        obs_encode = self.obs_encoder(obs)
        if len(next_obs.shape) == 3:
            next_obs = torch.unsqueeze(next_obs, 1)
        next_obs_encode = self.next_obs_encode(next_obs)
        obs_inter = torch.cat((obs_encode, next_obs_encode), dim=-1)

        z_mean, z_log_var = self.posterior_z(obs_inter)

        z = reparameterize(z_mean, z_log_var)

        if eval:
            z = torch.empty((batch_size, 64)).normal_(mean=0, std=1).cuda()

        kl_loss_z = -0.5 * (1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))
        kl_loss_z = torch.mean(kl_loss_z)

        return kl_loss_z, (z_mean, z_log_var), z_mean


class TaskCVAEModel(nn.Module):
    def __init__(self):
        super(TaskCVAEModel, self).__init__()
        self.posterior_z = LatentEncode(latent_dim=64, visual_encode_len=128, x_dim=128, hidden_dim=128)  # p(z|I, o)

    def forward(self, obs, next_obs, obs_encoder, eval=False):
        batch_size = obs.shape[0]
        obs_encode = obs_encoder(obs)
        if len(next_obs.shape) == 3:
            next_obs = torch.unsqueeze(next_obs, 1)
        next_obs_encode = obs_encoder(next_obs)
        tau = torch.cat((obs_encode, next_obs_encode), dim=-1)

        z_mean, z_log_var = self.posterior_z.forward(tau)

        z = reparameterize(z_mean, z_log_var)

        if eval:
            z = torch.empty((batch_size, 64)).normal_(mean=0, std=1).cuda()

        kl_loss_z = -0.5 * (1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))
        kl_loss_z = torch.mean(kl_loss_z)

        return kl_loss_z, (z_mean, z_log_var), z


class CPCModel(nn.Module):
    def __init__(self, skill_dim=64, visual_encode_len=128, hidden_dim=128):
        super().__init__()
        self.skill_dim = skill_dim
        self.state_dim = visual_encode_len
        self.temp = 0.5

        self.task_net = nn.Sequential(nn.Linear(self.state_dim * 2, hidden_dim), nn.LeakyReLU(0.1),
                                      nn.Linear(hidden_dim, skill_dim), nn.LeakyReLU(0.1))

        self.skill_net = nn.Sequential(nn.Linear(self.state_dim + self.skill_dim, hidden_dim), nn.LeakyReLU(0.1),
                                           nn.Linear(hidden_dim, skill_dim), nn.LeakyReLU(0.1))

        self.project_net = nn.Sequential(nn.Linear(self.skill_dim, hidden_dim), nn.LeakyReLU(0.1),
                                           nn.Linear(hidden_dim, skill_dim))

        self.trans = Transition()

    def forward(self, state, next_state, skill, state_net):
        assert len(state.size()) == len(next_state.size())
        state = state_net(state)
        next_state = state_net(next_state)
        task_feature = torch.cat([state, next_state], 1)

        task_feature = self.task_net(task_feature)
        skill = self.skill_net(torch.cat([state, skill], 1))

        trans_loss = self.trans.forward(state, next_state, skill)

        key = self.project_net(task_feature)
        query = self.project_net(skill)

        return query, key, skill, trans_loss


    def sample_skill(self, state, next_state, skill, state_net):
        assert len(state.size()) == len(next_state.size())
        state = state_net(state)
        next_state = state_net(next_state)
        task_feature = torch.cat([state, next_state], 1)

        task_feature = self.task_net(task_feature)
        skill = self.skill_net(torch.cat([state, skill], 1))

        return task_feature, skill


    def sample_skill_metric(self, state, next_state, skill, state_net):
        assert len(state.size()) == len(next_state.size())
        state = state_net.encode(state)
        next_state = state_net.encode(next_state)
        task_feature = state - next_state

        # task_feature = self.task_net(task_feature)
        skill = self.skill_net(torch.cat([state, skill], 1))

        return task_feature, skill


    def compute_cpc_loss(self, query, key):
        temperature = self.temp
        eps = 1e-6
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)
        cov = torch.mm(query, key.T)  # (b,b)
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)  # (b,)
        row_sub = torch.Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        pos = torch.exp(torch.sum(query * key, dim=-1) / temperature)  # (b,)
        loss = -torch.log(pos / (neg + eps))  # (b,)
        return loss, cov / temperature


class CURLModel(nn.Module):
    def __init__(self, skill_dim=64, visual_encode_len=128, hidden_dim=128):
        super().__init__()
        self.skill_dim = skill_dim
        self.state_dim = visual_encode_len


        self.task_net = nn.Sequential(nn.Linear(self.state_dim, hidden_dim), nn.ReLU(0.1),
                                      nn.Linear(hidden_dim, skill_dim), nn.ReLU(0.1), nn.LayerNorm(self.skill_dim))

        self.skill_net = nn.Sequential(nn.Linear(self.state_dim + self.skill_dim, hidden_dim), nn.ReLU(0.1),
                                           nn.Linear(hidden_dim, skill_dim), nn.ReLU(0.1), nn.LayerNorm(self.skill_dim))

        self.project_net = nn.Sequential(nn.Linear(self.skill_dim, hidden_dim), nn.LeakyReLU(0.1),
                                           nn.Linear(hidden_dim, skill_dim), nn.LayerNorm(self.skill_dim))

        self.trans = Transition()

        self.W = nn.Parameter(torch.rand(skill_dim, skill_dim))
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))



    def forward(self, state, next_state, skill, state_net):
        assert len(state.size()) == len(next_state.size())
        state = state_net(state)
        next_state = state_net(next_state)
        task_feature = torch.cat([state, next_state], 1)

        task_feature = self.task_net(task_feature)
        skill = self.skill_net(torch.cat([state, skill], 1))

        trans_loss = self.trans.forward(state, next_state, skill)

        key = self.project_net(task_feature)
        query = self.project_net(skill)

        return query, key, skill, trans_loss


    def forward_metric(self, state, next_state, skill, metric):
        assert len(state.size()) == len(next_state.size())
        state = metric.encode(state).detach()
        next_state = metric.encode(next_state).detach()
        task_feature = state - next_state
        task_label = self.compute_label(task_feature)

        task_feature = self.task_net(task_feature.detach())
        skill = self.skill_net(torch.cat([state, skill], 1))

        trans_loss = self.trans.forward(state, next_state, skill)

        # task_feature = self.project_net(task_feature)
        # skill = self.project_net(skill)

        return task_feature, skill, task_label, trans_loss


    def forward_clip(self, state, next_state, skill, metric):
        assert len(state.size()) == len(next_state.size())
        state = metric.encode(state).detach()
        next_state = metric.encode(next_state).detach()
        task_feature = state - next_state

        task_label = self.compute_label(task_feature)
        task_feature = self.task_net(task_feature.detach())
        skill = self.skill_net(torch.cat([state, skill], 1))

        trans_loss = self.trans.forward(state, next_state, skill)

        return task_feature, skill, task_label, trans_loss

    def compute_cluster_label(self, task):
        task_dist = torch.cdist(task, task, p=2)
        labels = torch.zeros_like(torch.arange(task.shape[0])).long().to(task.device)
        cluster_count = 0
        threshold = 0.1
        for i in range(1, labels.shape[0]):
            dist_vec = task_dist[i][:i]
            if dist_vec.min().item() < threshold:
                # same cluster
                min_index = torch.argmin(dist_vec).item()
                labels[i] = labels[min_index]
            else:
                cluster_count += 1
                labels[i] = cluster_count

        # labels = torch.arange(task.shape[0]).long().to(task.device)
        return labels

    def compute_label(self, task):
        # task_dist = torch.cdist(task, task, p=2)
        labels = torch.zeros_like(torch.arange(task.shape[0])).long().to(task.device)
        cluster_count = 0
        threshold = 0.1

        task_norm = torch.norm(task, dim=-1)

        for i in range(labels.shape[0]):
            if task_norm[i].item() < threshold:
                labels[i] = 0
            else:
                cluster_count += 1
                labels[i] = cluster_count
        return labels

    def sample_skill(self, state, next_state, skill, state_net):
        assert len(state.size()) == len(next_state.size())
        state = state_net(state)
        next_state = state_net(next_state)
        task_feature = torch.cat([state, next_state], 1)

        task_feature = self.task_net(task_feature)
        skill = self.skill_net(torch.cat([state, skill], 1))

        # task_feature = self.project_net(task_feature)
        # skill = self.project_net(skill)

        return task_feature, skill


    def sample_skill_metric(self, state, next_state, skill, state_net):
        assert len(state.size()) == len(next_state.size())
        state = state_net.encode(state)
        next_state = state_net.encode(next_state)
        task_feature = state - next_state

        task_feature = self.task_net(task_feature)
        skill = self.skill_net(torch.cat([state, skill], 1))

        # task_feature = self.project_net(task_feature)
        # skill = self.project_net(skill)

        return task_feature, skill


    def compute_curl_loss(self, query, key, labels):
        logits = self.compute_logits(query, key)
        # labels = torch.arange(logits.shape[0]).long().to(key.device)
        loss = self.cross_entropy_loss(logits, labels)

        return loss, logits


    def compute_logits(self, query, key):
        """
               Uses logits trick for CURL:
               - compute (B,B) matrix query (W key.T)
               - positives are all diagonal elements
               - negatives are all other elements
               - to compute loss use multiclass cross entropy with identity matrix for labels
        """

        wkey = torch.matmul(self.W, key.T)  # (z_dim,B)
        logits = torch.matmul(query, wkey)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]

        return logits


    def compute_clip_loss(self, task, skill, labels):
        # normalized features
        task_features = task / task.norm(dim=1, keepdim=True)
        skill_features = skill / skill.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_task = logit_scale * task_features @ skill_features.t()
        logits_per_skill = logits_per_task.t()

        # labels = torch.arange(logits_per_skill.shape[0]).long().to(skill.device)
        loss_task = self.cross_entropy_loss(logits_per_task, labels)
        loss_skill = self.cross_entropy_loss(logits_per_skill, labels)

        loss = 0.5 * (loss_task + loss_skill)

        # shape = [global_batch_size, global_batch_size]
        return loss, logits_per_skill


class Transition(nn.Module):
    def __init__(self, state_dim=128, skill_dim=64, hidden_dim=128, n_blocks=2):
        super().__init__()
        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_dim) for i in range(self.n_blocks)])

        self.obs_net = nn.Sequential(nn.Linear(state_dim + skill_dim, hidden_dim), nn.LeakyReLU(0.1))
        self.obs_predict = nn.Sequential(nn.Linear(hidden_dim, state_dim))

    def forward(self, state, next_state, skill):
        state_skill = torch.cat((state, skill), dim=1)
        next_state_predict = self.obs_net(state_skill)

        for i in range(self.n_blocks):
            next_state_predict = self.blocks[i](next_state_predict)

        next_state_predict = self.obs_predict(next_state_predict)
        next_state_predict = torch.sigmoid(next_state_predict)

        predict_loss = torch.norm(next_state_predict - next_state, dim=1)

        return predict_loss


# TODO this should be the CIC version
class TaskEncoder(nn.Module):
    def __init__(self):
        super(TaskCVAEModel, self).__init__()
        self.decoder = TestDecoder(latent_dim=256, visual_encode_len=128, hidden_dim1=128)  # p(I|z,o)
        self.obs_encoder = DepthEncoder(c_dim=128)
        self.next_obs_encode = DepthEncoder(c_dim=128, in_channel=1)
        self.posterior_z = LatentEncode(latent_dim=256, visual_encode_len=128, x_dim=128, hidden_dim=128)  # p(z|I, o)

    def forward(self, inter_params, obs, next_obs, eval=False):
        batch_size = obs.shape[0]
        obs_encode = self.obs_encoder(obs)
        if len(next_obs.shape) == 3:
            next_obs = torch.unsqueeze(next_obs, 1)
        next_obs_encode = self.next_obs_encode(next_obs)
        obs_inter = torch.cat((obs_encode, next_obs_encode), dim=-1)

        z_mean, z_log_var = self.posterior_z(obs_inter)

        z = reparameterize(z_mean, z_log_var)

        if eval:
            z = torch.empty((batch_size, 256)).normal_(mean=0, std=1).cuda()
        obs_z = torch.cat((obs_encode, z), dim=-1)
        recon_rotations, recon_point, recon_force = self.decoder(obs_z)

        kl_loss_z = -0.5 * (1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))
        kl_loss_z = torch.mean(kl_loss_z)

        return kl_loss_z, (z_mean, z_log_var), (recon_rotations, recon_point, recon_force), z


class ObsEncoder(nn.Module):
    """3D-convolutional encoder network for pixel input.

    Args:
        dim (int): input dimension
        c_dim (int): output dimension
    """

    def __init__(self, dim=3, c_dim=128):
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

class FinetuneEncode2(nn.Module):
    def __init__(self, latent_dim=64, visual_encode_len=128):
        super().__init__()
        self.actvn = F.relu
        self.fc_1 = nn.Linear(visual_encode_len, visual_encode_len)
        self.fc_2 = nn.Linear(visual_encode_len, latent_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        inputs = x
        net = self.actvn(self.fc_1(inputs))
        net = torch.sigmoid(self.fc_2(net))

        return net


class TaskCVAEModel2(nn.Module):
    def __init__(self):
        super(TaskCVAEModel2, self).__init__()
        self.latent_dim = 64
        self.visual_encode_len = 128

        self.posterior_z = TaskLatentEncode(latent_dim=self.latent_dim, visual_encode_len=self.visual_encode_len, hidden_dim=128)  # p(z|I, o)
        self.recon_model = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(0.1),
                                         nn.Linear(self.latent_dim, self.visual_encode_len), nn.Sigmoid()
                                         )

        self.condition_encode = ConditionEncode(latent_dim=self.latent_dim, visual_encode_len=self.visual_encode_len)
        self.finetune_encode = FinetuneEncode(latent_dim=self.latent_dim, visual_encode_len=self.visual_encode_len)
        # self.obs_encode = ObsEncoder(dim=3, c_dim=self.latent_dim)
        # self.finetune_encode2 = FinetuneEncode2(latent_dim=self.latent_dim, visual_encode_len=self.visual_encode_len)
        self.recon_loss = torch.nn.MSELoss()

    def forward(self, o0, o1, metric, finetune=False):
        o0_encode = metric.encode(o0)
        o1_encode = metric.encode(o1)
        o0_encode = o0_encode.detach()
        o1_encode = o1_encode.detach()

        if not finetune:
            tau = torch.cat((o0_encode, o1_encode), dim=-1)
            # tau = o0_encode - o1_encode
            z_mean, z_log_var = self.posterior_z.forward(tau)
            z_hat = reparameterize(z_mean, z_log_var)
            z = self.condition_encode.forward(z_hat, o0_encode)
            kl_loss_z = -0.5 * (1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))
            kl_loss_z = torch.mean(kl_loss_z)
        else:
            tau = o0_encode - o1_encode
            z_hat = self.finetune_encode.forward(tau)
            z = self.condition_encode.forward(z_hat, o0_encode)
            z_mean = None
            z_log_var = None
            kl_loss_z = None

        # recon_input = torch.cat((o0_encode, z), dim=-1)

        recon_o1 = self.recon_model(z)
        recon_loss = self.recon_loss(o1_encode, recon_o1)

        return kl_loss_z, (z_mean, z_log_var), z, recon_loss

    def inference(self, o0, o1, metric, noise, finetune=False):
        batch_size = o0.shape[0]
        o0_encode = metric.encode(o0)
        o1_encode = metric.encode(o1)
        o0_encode = o0_encode.detach()
        o1_encode = o1_encode.detach()

        tau = torch.cat((o0_encode, o1_encode), dim=-1)

        if not finetune:
            z = self.condition_encode.forward(noise, o0_encode)
        else:
            tau = o0_encode - o1_encode
            z = self.finetune_encode.forward(tau)
            z = self.condition_encode.forward(z, o0_encode)

        # recon_input = torch.cat((o0_encode, z), dim=-1)

        recon_o1 = self.recon_model(z)
        recon_loss = self.recon_loss(o1_encode, recon_o1)

        return z, recon_loss





class TaskGMCVAEModel(nn.Module):
    def __init__(self, category):
        super(TaskGMCVAEModel, self).__init__()
        self.latent_dim = 64
        self.visual_encode_len = 128
        self.y_dim = self.latent_dim // 2
        self.num_class = category
        self.w_rec = 1
        self.w_gauss= 2
        self.w_cat = 1

        self.cgmvae = CGMVAENet(self.visual_encode_len, self.visual_encode_len, self.y_dim, self.num_class)
        self.losses = LossFunctions()
        self.recon_loss = torch.nn.MSELoss()

    def forward(self, o0, o1, metric):
        batch_size = o0.shape[0]
        o0_encode = metric.encode(o0)
        o1_encode = metric.encode(o1)
        o0_encode = o0_encode.detach()
        o1_encode = o1_encode.detach()

        tau = o0_encode - o1_encode

        out_net = self.cgmvae.forward(tau, o0_encode)

        z, o1_recon, skill = out_net['gaussian'], out_net['x_rec'], out_net['skill']
        logits, prob_cat = out_net['logits'], out_net['prob_cat']
        y_mu, y_var = out_net['y_mean'], out_net['y_var']
        mu, var = out_net['mean'], out_net['var']

        # reconstruction loss
        loss_rec = self.recon_loss(o1_encode, o1_recon)

        # gaussian loss
        loss_gauss = self.losses.gaussian_loss(z, mu, var, y_mu, y_var)

        # categorical loss
        loss_cat = -self.losses.entropy(logits, prob_cat) - np.log(0.1)

        # total loss
        kl_loss = self.w_gauss * loss_gauss + self.w_cat * loss_cat
        trans_loss = self.w_rec * loss_rec

        return kl_loss, out_net, skill, trans_loss

    def inference(self, o0, metric, noise, category):
        o0_encode = metric.encode(o0)
        o0_encode = o0_encode.detach()

        if isinstance(category, list):
            assert len(category) == noise.shape[0]
            skill = torch.Tensor().to(noise.device)
            for i in range(len(category)):
                curr_o0_encode = o0_encode[i].unsqueeze(0)
                curr_noise = noise[i].unsqueeze(0)
                curr_skill = self.cgmvae.inference(curr_o0_encode, curr_noise, category[i], self.num_class)
                skill = torch.cat((curr_skill, skill), dim=0)
        else:
            skill = self.cgmvae.inference(o0_encode, noise, category, self.num_class)
        return skill
