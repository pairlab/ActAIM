import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import pdb

from new_scripts.model.task.multi_view_encoder import MultiViewEncoder
from affordance.utils.gmvae import CGMVAENet
from affordance.utils.LossFunctions import LossFunctions
from new_scripts.model.task.multi_view_encoder import ResnetBlockFC

class GmvaeTaskModel(nn.Module):
    def __init__(
        self,
        latent_dim=512,
    ):
        super(GmvaeTaskModel, self).__init__()


        self.multi_view_encode = MultiViewEncoder(latent_dim)
        self.visual_encode_len = 640
        self.y_dim = latent_dim // 2
        self.num_class = 10

        self.cgmvae = CGMVAENet(self.visual_encode_len, self.visual_encode_len, self.y_dim, self.num_class)
        self.w_rec = 1
        self.w_gauss= 2
        self.w_cat = 1

        self.softmax = nn.Softmax(dim=1)
        self.losses = LossFunctions()
        self.recon_loss = torch.nn.MSELoss()

        self.cond_decode = ResnetBlockFC(self.visual_encode_len * 2, self.visual_encode_len)


    def forward(self, curr_obs, final_obs):
        bs = curr_obs.shape[0]

        curr_x = self.multi_view_encode(curr_obs).view(bs, -1)
        final_x = self.multi_view_encode(final_obs).view(bs, -1)

        # define task representation as the difference between initial and final obs
        task_label = final_x - curr_x
        out_net = self.cgmvae.forward(task_label, curr_x)

        z, final_x_predict, skill = out_net['gaussian'], out_net['x_rec'], out_net['skill']
        logits, prob_cat = out_net['logits'], out_net['prob_cat']
        y_mu, y_var = out_net['y_mean'], out_net['y_var']
        mu, var = out_net['mean'], out_net['var']

        # reconstruction loss
        loss_rec = self.recon_loss(final_x, final_x_predict)

        # gaussian loss
        loss_gauss = self.losses.gaussian_loss(z, mu, var, y_mu, y_var)

        # categorical loss
        loss_cat = -self.losses.entropy(logits, prob_cat) - np.log(0.1)

        # total loss
        kl_loss = self.w_gauss * loss_gauss + self.w_cat * loss_cat
        trans_loss = self.w_rec * loss_rec

        total_loss = kl_loss + trans_loss

        embed_cond_x = torch.cat((skill, curr_x), dim=-1)
        task_embed = self.cond_decode(embed_cond_x)


        return total_loss.mean(), task_embed, None, task_label, final_x

    def sample(self, curr_obs, device, task_embed_ind=None):
        curr_x = self.multi_view_encode(curr_obs)
        quantized, select_ind = self.vq.sample(curr_x, device)

        return quantized, select_ind