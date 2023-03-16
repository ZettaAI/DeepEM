from __future__ import annotations

from functools import partial

import numpy as np
import torch


class BinaryWeightBalancer():
    """
    Computes a weight map by balancing foreground/background.
    """

    def __init__(
        self,
        weight0: float | None,
        weight1: float | None,
        clipmin: float = 0.01,
        clipmax: float = 0.99,
    ):
        assert weight0 > 0 if weight0 is not None else True
        assert weight1 > 0 if weight1 is not None else True
        self.weight0 = weight0
        self.weight1 = weight1
        self.dynamic = (weight0 is None) and (weight1 is None)
        self.clip = partial(np.clip, a_min=clipmin, a_max=clipmax)

    def __call__(
        self,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        dtype = mask.dtype
        ones = mask * torch.eq(target, 1).type(dtype)
        zeros = mask * torch.eq(target, 0).type(dtype)

        # Dynamic balancing
        if self.dynamic:

            n_ones = ones.sum().item()
            n_zeros = zeros.sum().item()
            if (n_ones + n_zeros) > 0:
                ones *= self.clip(n_zeros / (n_ones + n_zeros))
                zeros *= self.clip(n_ones / (n_ones + n_zeros))

        # Static balancing
        else:

            if self.weight1 is not None:
                ones *= self.weight1

            if self.weight0 is not None:
                zeros *= self.weight0

        return (ones + zeros).type(dtype)
    