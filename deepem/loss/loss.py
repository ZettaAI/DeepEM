import numpy as np

import torch
from torch import nn
from torch.nn import functional as F


class BCELoss(nn.Module):
    """
    Binary cross entropy loss with logits.
    """
    def __init__(self, size_average=True, margin0=0, margin1=0, inverse=True,
                       class_balancer=None, **kwargs):
        super().__init__()
        self.bce = F.binary_cross_entropy_with_logits
        self.size_average = size_average
        self.margin0 = float(np.clip(margin0, 0, 1))
        self.margin1 = float(np.clip(margin1, 0, 1))
        self.inverse = inverse
        self.balancer = class_balancer

    def forward(self, input, target, mask):
        # Number of valid voxels
        nmsk = (mask > 0).to(dtype=mask.dtype).sum()
        assert nmsk.item() >= 0
        if nmsk.item() == 0:
            # Return a graph-connected zero on the right device/dtype
            zero = (input * mask).sum()
            return zero, nmsk

        # Class balancing
        if self.balancer is not None:
            mask = self.balancer(target, mask)

        # Margin
        tgt = target
        m0, m1 = self.margin0, self.margin1
        if m0 > 0 or m1 > 0:
            if self.inverse:
                tgt = target.clone()
                tgt[torch.eq(target, 1)] = 1 - m1
                tgt[torch.eq(target, 0)] = m0
            else:
                activ = torch.sigmoid(input)
                m_int = torch.ge(activ, 1 - m1) * torch.eq(target, 1)
                m_ext = torch.le(activ, m0) * torch.eq(target, 0)
                mask *= 1 - (m_int + m_ext).type(mask.dtype)

        loss = self.bce(input, tgt, weight=mask, reduction='sum')

        if self.size_average:
            loss = loss / nmsk.item()
            nmsk = torch.tensor(1, dtype=nmsk.dtype, device=nmsk.device)

        return loss, nmsk


class MSELoss(nn.Module):
    """
    Mean squared error loss with (or without) logits.
    """
    def __init__(self, size_average=True, margin0=0, margin1=0, logits=True,
                       class_balancer=None, **kwargs):
        super().__init__()
        self.mse = F.mse_loss
        self.size_average = size_average
        self.margin0 = float(np.clip(margin0, 0, 1))
        self.margin1 = float(np.clip(margin1, 0, 1))
        self.logits = logits
        self.balancer = class_balancer

    def forward(self, input, target, mask):
        # Number of valid voxels
        nmsk = (mask > 0).to(dtype=mask.dtype).sum()
        assert nmsk.item() >= 0
        if nmsk.item() == 0:
            # Return a graph-connected zero on the right device/dtype
            zero = (input * mask).sum()
            return zero, nmsk

        # Class balancing
        if self.balancer is not None:
            mask = self.balancer(target, mask)

        activ = torch.sigmoid(input) if self.logits else input

        # Margin
        m0, m1 = self.margin0, self.margin1
        if m0 > 0 or m1 > 0:
            m_int = torch.ge(activ, 1 - m1) * torch.eq(target, 1)
            m_ext = torch.le(activ, m0) * torch.eq(target, 0)
            mask *= 1 - (m_int + m_ext).type(mask.dtype)

        loss = self.mse(activ, target, reduction='none')
        loss = (loss * mask).sum()

        if self.size_average:
            loss = loss / nmsk.item()
            nmsk = torch.tensor(1, dtype=nmsk.dtype, device=nmsk.device)

        return loss, nmsk
