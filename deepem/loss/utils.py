from __future__ import annotations

from functools import partial

import numpy as np
import torch


class BinaryWeightBalancer():
    """
    Computes a weight map by balancing foreground/background.
    Supports both scalar and directional (per-channel) weights.
    """

    def __init__(
        self,
        weight0: float | list[float] | None,
        weight1: float | list[float] | None,
        clipmin: float = 0.01,
        clipmax: float = 0.99,
    ):
        # Validate and store weights
        self.weight0 = self._validate_weight(weight0)
        self.weight1 = self._validate_weight(weight1)

        # Determine if weights are directional
        self.directional = isinstance(self.weight0, list) or isinstance(self.weight1, list)

        # Broadcast scalar to list if mixed with directional
        if self.directional:
            if isinstance(self.weight0, (int, float)):
                self.weight0 = [self.weight0] * 3
            if isinstance(self.weight1, (int, float)):
                self.weight1 = [self.weight1] * 3

        self.dynamic = (weight0 is None) and (weight1 is None)
        self.clip = partial(np.clip, a_min=clipmin, a_max=clipmax)

    def _validate_weight(self, weight):
        """Validate weight is positive if provided."""
        if weight is None:
            return None
        if isinstance(weight, list):
            assert len(weight) == 3, f"Directional weights must have 3 values, got {len(weight)}"
            for w in weight:
                assert w > 0, f"Weight must be positive, got {w}"
            return weight
        else:
            assert weight > 0, f"Weight must be positive, got {weight}"
            return weight

    def __call__(
        self,
        target: torch.Tensor,
        mask: torch.Tensor,
        channel: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            target: Target tensor
            mask: Mask tensor
            channel: Optional channel index for directional weights (0=x, 1=y, 2=z)

        Returns:
            Weighted mask tensor
        """
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
            # Get weights for this channel
            if self.directional and channel is not None:
                w0 = self.weight0[channel] if self.weight0 is not None else None
                w1 = self.weight1[channel] if self.weight1 is not None else None
            else:
                # Backward compatibility: use scalar or first element
                w0 = self.weight0 if not isinstance(self.weight0, list) else self.weight0[0]
                w1 = self.weight1 if not isinstance(self.weight1, list) else self.weight1[0]

            if w1 is not None:
                ones *= w1

            if w0 is not None:
                zeros *= w0

        return (ones + zeros).type(dtype)
    