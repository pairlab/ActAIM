import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os
import numpy as np

import pdb


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    """Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class MultiViewEncoder(nn.Module):
    """
    Args:
      full_homogeneous: if True, all convs are on homogeneous space; else only first is.
    """

    def __init__(self, latent_dim=512):
        super(MultiViewEncoder, self).__init__()

        # Use VGG to encode
        self.vgg_model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True)
        self.vgg_model.eval()

        # Remove the final fully connected layers of the network
        self.vgg_model.classifier = self.vgg_model.classifier[:-3]

        # Freeze the parameters of the feature extractor
        for param in self.vgg_model.parameters():
            param.requires_grad = False

        self.max_pool = nn.MaxPool1d(32, stride=32)

        # vgg_fc
        # 4096 vgg feature dim
        # self.filter_nn = torch.nn.ModuleList([
        #     ResnetBlockFC(4096, 128, 1024),
        #     nn.ReLU(),
        #     ResnetBlockFC(128, 16, 64),
        # ])

        self.activation = nn.Sigmoid()


    # def filter(self, x):
    #     for layer in self.filter_nn:
    #         x = layer(x)
    #     return x


    def forward(self, x):

        shp = x.shape
        x = x.view((shp[0] * shp[1], *shp[2:]))

        x = self.vgg_model(x).detach()

        # PCA slows down the training process
        # PCA in torch
        # mean_centered_tensor = x - x.mean(dim=0, keepdim=True)
        # covariance_matrix = torch.mm(mean_centered_tensor.t(), mean_centered_tensor) / (x.size(0) - 1)
        # U, S, V = torch.svd(covariance_matrix)
        # output_dim = 128
        # x = U[:, :output_dim]
        # x = torch.mm(mean_centered_tensor, x)
        # x = self.filter(x)

        # batch, views, ...
        x = x.view((shp[0], shp[1], x.shape[1]))
        x = self.max_pool(x)
        x = x.view((shp[0] * shp[1], -1))

        x = self.activation(x)
        x = x.view((shp[0], shp[1], x.shape[1]))

        return x.detach()