from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import Function
import torch.backends.cudnn as cudnn


class L2dist(Function):
    def __init__(self, p):
        super(L2dist, self).__init__()
        self.norm = p

    def forward(self, x1, x2):
        eps = 1e-4 / x1.size(0)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, self.norm).sum(dim=1)
        return torch.pow(out + eps, 1.0 / self.norm)


class LogRatioLoss(Function):
    """Log ratio loss function."""

    def __init__(self):
        super(LogRatioLoss, self).__init__()
        self.pdist = L2dist(2)  # norm 2

    def forward(self, input, gt_dist):
        m = input.size()[0] - 1  # #paired
        a = input[0]  # anchor
        p = input[1:]  # paired

        # auxiliary variables
        idxs = torch.arange(1, m + 1).cuda()
        indc = idxs.repeat(m, 1).t() < idxs.repeat(m, 1)

        epsilon = 1e-6

        dist = self.pdist.forward(a, p)
        gt_dist = gt_dist[0][1:]
        log_dist = torch.log(dist + epsilon)
        log_gt_dist = torch.log(gt_dist + epsilon)
        diff_log_dist = log_dist.repeat(m, 1).t() - log_dist.repeat(m, 1)
        diff_log_gt_dist = log_gt_dist.repeat(m, 1).t() - log_gt_dist.repeat(m, 1)

        # uniform weight coefficients
        wgt = indc.clone().float()
        wgt = wgt.div(wgt.sum())

        log_ratio_loss = (diff_log_dist - diff_log_gt_dist).pow(2)

        loss = log_ratio_loss
        loss = loss.mul(wgt).sum()

        return loss


class MarginLoss(Function):
    def __init__(self, margin=0.4, modes=False):
        super(MarginLoss, self).__init__()
        self.margin = margin
        self.modes = modes

    def forward(self, input, gt_dist):
        input_dist = torch.cdist(input, input)
        if self.modes:
            negative_pair_loss = torch.where(
                self.margin - input_dist < 0, torch.zeros_like(input_dist), self.margin - input_dist
            )
            loss = torch.where(gt_dist > 0, negative_pair_loss, input_dist)
        else:
            loss = torch.where(gt_dist > self.margin, torch.zeros_like(input_dist), input_dist)
        return loss.flatten().sum()
