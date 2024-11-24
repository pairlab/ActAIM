import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import pdb

from new_scripts.model.task.multi_view_encoder import MultiViewEncoder

class GoalTaskModel(nn.Module):
    def __init__(
        self,
        latent_dim=512,
    ):
        super(GoalTaskModel, self).__init__()


        self.multi_view_encode = MultiViewEncoder(latent_dim)
        self.bn1 = nn.BatchNorm1d(5)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, curr_obs, final_obs, dof=None):
        bs = curr_obs.shape[0]

        curr_x = self.multi_view_encode(curr_obs)
        final_x = self.multi_view_encode(final_obs)

        # define task representation as the difference between initial and final obs
        # loss, quantized, perplexity, task_label = self.vq(curr_x, final_x)

        # task_label = task_label.view(quantized.shape)
        # quantized = self.softmax(quantized)

        # task_loss, task_embed, task_label
        task_embed = curr_x - final_x
        task_embed = self.bn1(task_embed)
        task_embed = task_embed.view(bs, -1)

        return 0, task_embed, task_embed, final_x.view(bs, -1)


    def get_task_embed(self, curr_obs, final_obs):
        bs = curr_obs.shape[0]

        curr_x = self.multi_view_encode(curr_obs)
        final_x = self.multi_view_encode(final_obs)

        # task_loss, task_embed, task_label
        task_embed = curr_x - final_x
        task_embed = self.bn1(task_embed)
        task_embed = task_embed.view(bs, -1)

        return task_embed

class ZeroTaskModel(nn.Module):
    def __init__(
        self,
        latent_dim=512,
    ):
        super(ZeroTaskModel, self).__init__()


        self.multi_view_encode = MultiViewEncoder(latent_dim)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, curr_obs, final_obs, dof):
        bs = curr_obs.shape[0]
        # curr_x = self.multi_view_encode(curr_obs)
        final_x = self.multi_view_encode(final_obs)

        # define task representation as the difference between initial and final obs
        # loss, quantized, perplexity, task_label = self.vq(curr_x, final_x)

        # task_label = task_label.view(quantized.shape)
        # quantized = self.softmax(quantized)

        # task_loss, task_embed, task_label
        task_embed = torch.zeros_like(dof)
        return 0, task_embed, task_embed, final_x.view(bs, -1)


class GTaskModel(nn.Module):
    def __init__(
        self,
        latent_dim=512,
    ):
        super(GTaskModel, self).__init__()


        self.multi_view_encode = MultiViewEncoder(latent_dim)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, curr_obs, final_obs, dof):
        # curr_x = self.multi_view_encode(curr_obs)
        final_x = self.multi_view_encode(final_obs)

        # define task representation as the difference between initial and final obs
        # loss, quantized, perplexity, task_label = self.vq(curr_x, final_x)

        # task_label = task_label.view(quantized.shape)
        # quantized = self.softmax(quantized)

        # task_loss, task_embed, task_label
        return 0, dof, dof, final_x.view(bs, -1)



    def sample(self, curr_obs, device):
        curr_x = self.multi_view_encode(curr_obs)
        # select_ind is None

        return quantized, None