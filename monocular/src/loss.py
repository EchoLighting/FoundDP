import time
import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F


class SILogLoss(nn.Module):
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=False):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]

        g = torch.log(input) - torch.log(target)

        # Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        if g.numel() > 1:
            Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        else:
            Dg = torch.pow(torch.mean(g), 2)

        return 10 * torch.sqrt(Dg)


class CharbonnierLoss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(CharbonnierLoss, self).__init__()
        self.name = 'Charbonnier'
        self.eps = 1e-4

    def forward(self, input, target, mask):
        if mask is not None:
            input = input[mask]
            target = target[mask]
        diff = torch.add(input, -target)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

