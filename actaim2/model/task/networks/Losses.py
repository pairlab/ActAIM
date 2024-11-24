# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Loss functions used for training our model

"""
import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import pdb


class LossFunctions:
    eps = 1e-8

    def mean_squared_error(self, real, predictions):
        """Mean Squared Error between the true and predicted outputs
           loss = (1/n)*Σ(real - predicted)^2

        Args:
            real: (array) corresponding array containing the true labels
            predictions: (array) corresponding array containing the predicted labels

        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                                  of all the sample losses or an array with the losses per sample
        """
        loss = (real - predictions).pow(2)
        return loss.sum(-1).mean()

    def reconstruction_loss(self, real, predicted, rec_type='mse'):
        """Reconstruction loss between the true and predicted outputs
           mse = (1/n)*Σ(real - predicted)^2
           bce = (1/n) * -Σ(real*log(predicted) + (1 - real)*log(1 - predicted))

        Args:
            real: (array) corresponding array containing the true labels
            predictions: (array) corresponding array containing the predicted labels

        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                                  of all the sample losses or an array with the losses per sample
        """
        if rec_type == 'mse':
            loss = (real - predicted).pow(2)
        elif rec_type == 'bce':
            loss = F.binary_cross_entropy(predicted, real, reduction='none')
        else:
            raise "invalid loss function... try bce or mse..."
        return loss.sum(-1).mean()

    def log_normal(self, x, mu, var):
        """Logarithm of normal distribution with mean=mu and variance=var
           log(x|μ, σ^2) = loss = -0.5 * Σ log(2π) + log(σ^2) + ((x - μ)/σ)^2

        Args:
           x: (array) corresponding array containing the input
           mu: (array) corresponding array containing the mean
           var: (array) corresponding array containing the variance

        Returns:
           output: (array/float) depending on average parameters the result will be the mean
                                  of all the sample losses or an array with the losses per sample
        """
        if self.eps > 0.0:
            var = var + self.eps
        return -0.5 * torch.sum(
            np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1)

    def gaussian_loss(self, z, z_mu, z_var, z_mu_prior, z_var_prior):
        """Variational loss when using labeled data without considering reconstruction loss
           loss = log q(z|x,y) - log p(z) - log p(y)

        Args:
           z: (array) array containing the gaussian latent variable
           z_mu: (array) array containing the mean of the inference model
           z_var: (array) array containing the variance of the inference model
           z_mu_prior: (array) array containing the prior mean of the generative model
           z_var_prior: (array) array containing the prior variance of the generative mode

        Returns:
           output: (array/float) depending on average parameters the result will be the mean
                                  of all the sample losses or an array with the losses per sample
        """
        loss = self.log_normal(z, z_mu, z_var) - self.log_normal(z, z_mu_prior, z_var_prior)
        return loss.mean()

    def entropy(self, logits, targets):
        """Entropy loss
            loss = (1/n) * -Σ targets*log(predicted)

        Args:
            logits: (array) corresponding array containing the logits of the categorical variable
            real: (array) corresponding array containing the true labels

        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                                  of all the sample losses or an array with the losses per sample
        """
        log_q = F.log_softmax(logits, dim=-1)
        return -torch.mean(torch.sum(targets * log_q, dim=-1))

    def contrastive_loss(self, embeddings, labels):
        # Calculate pairwise distance between embeddings
        pairwise_distance = torch.cdist(embeddings, embeddings, p=2)

        # Create mask to identify pairs with same cluster label
        label_mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()

        # Create mask to identify pairs with different cluster labels
        diff_label_mask = 1 - label_mask

        # Calculate loss for pairs with same cluster label
        loss_same = torch.sum(label_mask * torch.pow(pairwise_distance, 2)) / torch.sum(label_mask)

        self.margin = 2.0

        # Calculate loss for pairs with different cluster labels
        loss_diff = torch.sum(
            diff_label_mask * torch.pow(torch.clamp(self.margin - pairwise_distance, min=0.0), 2)) / torch.sum(
            diff_label_mask)

        # Total loss is the sum of losses for same and different cluster labels
        total_loss = loss_same + loss_diff

        return total_loss