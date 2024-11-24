from __future__ import print_function


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


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

import pdb


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings=8, embedding_dim=128*5, commitment_cost=10.0, decay=0.8, epsilon=1e-5, temperature=1.0):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon
        self.temperature = temperature

        self.cond_decode = ResnetBlockFC(self._embedding_dim * 2, self._embedding_dim)
        self.beta = 0.25
        self.recon_weight = 1.0

        self.mse_loss = nn.MSELoss(reduce=False, reduction='mean')
        self.cos_sim_loss = nn.CosineEmbeddingLoss(reduce=False, reduction='mean')

        # add KL loss here
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, cond_x, inputs):
        # use vector representation
        # convert inputs from BCHW -> BHWC
        # inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        bs = input_shape[0]

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Conditional decode

        recon_loss = torch.mean((quantized.detach()-inputs)**2) + self.beta * \
            torch.mean((quantized - inputs.detach()) ** 2)

        quantized_flat = quantized.view(-1, self._embedding_dim)
        cond_x_flat = cond_x.view(-1, self._embedding_dim)

        embed_cond_x = torch.cat((quantized_flat, cond_x_flat), dim=-1)
        task_embed = self.cond_decode(embed_cond_x)

        task_embed = task_embed.view(input_shape)

        # loss
        task_label = inputs - cond_x

        # TODO compute KL div loss instead
        # TODO CLIP requires large batch size
        # task_embed_4loss = F.log_softmax(task_embed, dim=1)
        # task_label_4loss = F.softmax(task_label, dim=1)

        # e_latent_loss = self.compute_clip_loss(task_embed, task_label.detach())
        # e_latent_loss = self.kl_loss(task_embed_4loss, task_label_4loss.detach())

        # task_embed_4loss = task_embed.view(bs, -1)
        # task_label_4loss = task_label.view(bs, -1)
        # target = torch.ones(bs).to(inputs.device)
        # e_latent_loss = self.cos_sim_loss(task_embed_4loss, task_label_4loss.detach(), target)

        e_latent_loss = self.mse_loss(task_embed, task_label.detach())
        loss = self._commitment_cost * e_latent_loss + self.recon_weight * recon_loss

        # Straight Through Estimator
        inputs = inputs.view(bs, -1)
        task_embed = task_embed.view(bs, -1)
        task_label = task_label.view(bs, -1)
        task_embed = inputs + (task_embed - inputs).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        # return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
        return loss, task_embed.contiguous(), perplexity, task_label

    def test_task_embed(self, cond_x, inputs):
        # use vector representation
        # convert inputs from BCHW -> BHWC
        # inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        bs = input_shape[0]

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Conditional decode

        recon_loss = torch.mean((quantized.detach()-inputs)**2) + self.beta * \
            torch.mean((quantized - inputs.detach()) ** 2)

        quantized_flat = quantized.view(-1, self._embedding_dim)
        cond_x_flat = cond_x.view(-1, self._embedding_dim)

        embed_cond_x = torch.cat((quantized_flat, cond_x_flat), dim=-1)
        task_embed = self.cond_decode(embed_cond_x)

        task_embed = task_embed.view(input_shape)

        # loss
        task_label = inputs - cond_x
        e_latent_loss = self.compute_clip_loss(task_embed, task_label.detach())
        loss = self._commitment_cost * e_latent_loss + self.recon_weight * recon_loss

        # Straight Through Estimator
        inputs = inputs.view(bs, -1)
        task_embed = task_embed.view(bs, -1)
        task_label = task_label.view(bs, -1)
        task_embed = inputs + (task_embed - inputs).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        # return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
        return loss, self._commitment_cost * e_latent_loss, self.recon_weight * recon_loss, task_embed.contiguous(), quantized, task_label


    def sample(self, cond_x, task_embed_ind=None, device="cuda"):
        cond_x_shp = cond_x.shape
        cond_x_flat = cond_x.view(-1, self._embedding_dim)
        num_samples = cond_x_flat.shape[0]

        with torch.no_grad():
            codebook = self._embedding
            embedding_dim = self._embedding_dim

            samples = []
            for _ in range(num_samples):
                # Initialize a random tensor
                quantized = torch.randn(1, embedding_dim).to(device)
                encoding_indices = torch.randint(0, codebook.num_embeddings, (1,)).to(device)
                if task_embed_ind is not None:
                    encoding_indices = torch.ones_like(encoding_indices).to(device) * task_embed_ind
                print("+++++++++++++++++++++++++++++++++current selected codebook index: ", encoding_indices.item())
                quantized = codebook(encoding_indices).unsqueeze(0)
                samples.append(quantized)

            samples = torch.stack(samples, dim=0)

            samples_flat = samples.view(-1, self._embedding_dim)

            embed_cond_x = torch.cat((samples_flat, cond_x_flat), dim=-1)
            task_embed = self.cond_decode(embed_cond_x)
            task_embed = task_embed.view((cond_x_shp[0], -1))

        return task_embed, encoding_indices.item()


    def compute_clip_loss(self, task_latent, task_label):
        assert task_latent.shape == task_label.shape
        if len(task_latent.shape) > 2:
            bs = task_label.shape[0]
            task_latent = task_latent.view(bs, -1)
            task_label = task_label.view(bs, -1)


        # Calculating the Loss
        logits = (task_latent @ task_label.T) / self.temperature
        latent_similarity = task_latent @ task_latent.T
        label_similarity = task_label @ task_label.T
        targets = F.softmax(
            (latent_similarity + label_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = self.cross_entropy(logits, targets, reduction='none')
        images_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
        return loss.mean()


    def cross_entropy(self, preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()


class TaskModel(nn.Module):
    def __init__(
        self,
        latent_dim=512,
    ):
        super(TaskModel, self).__init__()


        self.multi_view_encode = MultiViewEncoder(latent_dim)
        self.vq = VectorQuantizerEMA()

        self.softmax = nn.Softmax(dim=1)


    def forward(self, curr_obs, final_obs):
        bs = curr_obs.shape[0]
        curr_x = self.multi_view_encode(curr_obs)
        final_x = self.multi_view_encode(final_obs)

        # define task representation as the difference between initial and final obs
        loss, quantized, perplexity, task_label = self.vq(curr_x, final_x)
        task_label = task_label.view(quantized.shape)
        quantized = self.softmax(quantized)

        return loss.mean(), quantized, perplexity, task_label, final_x.view(bs, -1)

    def test_task_embed(self, curr_obs, final_obs):
        curr_x = self.multi_view_encode(curr_obs)
        final_x = self.multi_view_encode(final_obs)

        # define task representation as the difference between initial and final obs
        total_loss, clip_loss, vec_recon_loss, task_embed, quantized, task_label = self.vq.test_task_embed(curr_x, final_x)
        task_label = task_label.view(quantized.shape)

        return total_loss.mean(), clip_loss.mean(), vec_recon_loss.mean(), task_embed, quantized, task_label



    def sample(self, curr_obs, device, task_embed_ind=None):
        curr_x = self.multi_view_encode(curr_obs)
        quantized, select_ind = self.vq.sample(curr_x, device, task_embed_ind)

        return quantized, select_ind


class ImageDecoder(nn.Module):
    def __init__(self):
        super(ImageDecoder, self).__init__()

        self.fc = nn.Linear(128, 256)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0)
        self.tanh = nn.Tanh()

    def forward(self, features, img):
        bs = features.size(0)
        x = self.fc(features)

        x = self.relu(x)
        x = x.view(bs, 256, 1, 1)
        x = F.interpolate(x, scale_factor=3, mode='nearest')
        x = self.conv1(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=4, mode='nearest')
        x = self.conv2(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=5, mode='nearest')
        x = self.conv3(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv4(x)
        x = self.tanh(x)
        pdb.set_trace()

        output = self.tanh(x)


        pdb.set_trace()


