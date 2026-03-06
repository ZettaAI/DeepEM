from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.nn import functional as F

from deepem.utils import torch_utils


def create_mapping(trgt: npt.NDArray, splt: npt.NDArray, mask: npt.NDArray) -> list[list[int]]:
    trgt, splt = trgt.astype(np.uint64), splt.astype(np.uint64)
    encoded = (2 ** 32) * trgt + splt
    encoded[mask == 0] = 0
    unq = np.unique(encoded)
    mapping: dict[int, list[int]] = {}
    for unq_id in unq:
        trgt_id, splt_id = int(unq_id // (2 ** 32)), int(unq_id % (2 ** 32))
        mapping[trgt_id] = mapping.get(trgt_id, []) + [splt_id]
    result = list(mapping.values())
    return result


def compute_affinity(
    embd1: torch.Tensor,
    embd2: torch.Tensor,
    dim: int = -4,
    keepdims: bool = True,
    delta_d: float = 1.5,
) -> torch.Tensor:
    """Compute an affinity map from a pair of embeddings."""
    norm = torch.norm(embd1 - embd2, p=1, dim=dim, keepdim=keepdims)
    margin = (2 * delta_d - norm) / (2 * delta_d)
    zero = torch.zeros(1, dtype=embd1.dtype, device=embd1.device)
    result = torch.max(zero, margin) ** 2
    return result


def vec2aff(
    vec: torch.Tensor,
    edges: Sequence[Sequence[int]] = [(0, 0, 1),(0, 1, 0),(1, 0, 0)],
    delta_d: float = 1.5,
):
    assert vec.ndimension() >= 4
    assert len(edges) > 0

    affs = []
    for edge in edges:
        aff = compute_affinity(*(torch_utils.get_pair(vec, edge)), delta_d=delta_d)
        pad = ()
        for e in reversed(edge):
            if e > 0:
                pad += (e, 0)
            else:
                pad += (0, abs(e))
        affs.append(F.pad(aff, pad))

    assert len(affs) > 0
    for aff in affs:
        assert affs[0].size() == aff.size()

    return torch.cat(affs, dim=-4)


def _downsample_zero_padded(
    embd: torch.Tensor,
    trgt: torch.Tensor,
    mask: torch.Tensor,
    splt: torch.Tensor | None,
    scale_factor: tuple[float, float, float],
    sr_scale_z: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Downsample zero-padded aniso tensors without mixing real and padded values.

    Extracts real sections (at offset::sr_scale_z), then downsamples XY only.
    No re-padding: zero-padded positions have zero masks and don't contribute
    to the loss, so we just discard them.
    """
    offset = sr_scale_z // 2
    xy_scale = (1.0, scale_factor[1], scale_factor[2])

    def process(t: torch.Tensor, mode: str, **kwargs) -> torch.Tensor:
        t = t[:, :, offset::sr_scale_z, :, :]  # extract real sections
        return F.interpolate(t, scale_factor=xy_scale, mode=mode, **kwargs)

    embd = process(embd, mode='trilinear', align_corners=False)
    trgt = process(trgt, mode='nearest')
    mask = process(mask, mode='nearest')
    if splt is not None:
        splt = process(splt, mode='nearest')

    return embd, trgt, mask, splt


class MeanLoss(nn.Module):
    """
    Means-based loss for metric embeddings.
    """    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.001,
        delta_v: float = 0.0,
        delta_d: float = 1.5,
        recompute_ext: bool = False,
        mask_background: bool = True,
        loss_scale_factor: tuple[float, float, float] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta_v = delta_v  # Variance (intra-cluster pull force) hinge
        self.delta_d = delta_d  # Distance (inter-cluster push force) hinge
        self.recompute_ext = recompute_ext
        self.mask_background = mask_background
        self.loss_scale_factor = loss_scale_factor

    def forward(
        self,
        embd: torch.Tensor,
        trgt: torch.Tensor,
        mask: torch.Tensor,
        splt: torch.Tensor | None = None,
        sr_scale_z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        :param embd: Embeddings
        :param trgt: Target segmentation
        :param mask: Segmentation mask
        :param splt: Connected components of the target segmentation
        :param sr_scale_z: SR scale factor in Z (>0 for zero-padded aniso samples)
        """
        device = embd.device

        # Downsample if enabled
        if self.loss_scale_factor is not None:
            sz = int(sr_scale_z.item()) if sr_scale_z is not None else 0
            if sz > 0:
                embd, trgt, mask, splt = _downsample_zero_padded(
                    embd, trgt, mask, splt, self.loss_scale_factor, sz
                )
            else:
                embd = F.interpolate(embd, scale_factor=self.loss_scale_factor, mode='trilinear', align_corners=False)
                trgt = F.interpolate(trgt, scale_factor=self.loss_scale_factor, mode='nearest')
                mask = F.interpolate(mask, scale_factor=self.loss_scale_factor, mode='nearest')
                if splt is not None:
                    splt = F.interpolate(splt, scale_factor=self.loss_scale_factor, mode='nearest')

        groups = None
        if self.recompute_ext:
            assert splt is not None
            trgt = torch.squeeze(trgt)
            splt = torch.squeeze(splt)
            mask = torch.squeeze(mask)
            groups = create_mapping(trgt.cpu().numpy(), splt.cpu().numpy(), mask.cpu().numpy())
            trgt = splt

        trgt = trgt.to(torch.int)

        # Filter out background and get unique IDs
        masked_trgt = trgt[mask > 0]
        if self.mask_background:
            masked_trgt = masked_trgt[masked_trgt != 0]
        ids = torch.unique(masked_trgt).tolist()

        # Recompute external matrix
        mext = self.compute_ext_matrix(ids, groups, self.recompute_ext, device)
        vecs = self.generate_vecs(embd, trgt, mask, ids)
        means = [torch.mean(vec, dim=0) for vec in vecs]
        weights = [1.0] * len(vecs)

        # Dummy nmsk
        nmsk = torch.tensor([1]).to(device, dtype=torch.float)

        # Compute loss
        loss_int = self.compute_loss_int(vecs, means, weights, device)
        loss_ext = self.compute_loss_ext(means, weights, mext, device)
        loss_nrm = self.compute_loss_nrm(means, device)

        loss = (self.alpha * loss_int) + (self.beta * loss_ext) + (self.gamma * loss_nrm)
        # Only handle the empty case. Keep loss intact otherwise.
        if not vecs:
            # Graph-connected zero during training. Safe no-op during eval.
            loss = (embd * mask).sum() * 0
        return loss, nmsk

    def compute_loss_int(
        self,
        vecs: list[torch.Tensor],
        means: list[torch.Tensor],
        weights: list[float],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute the internal term of the loss."""
        assert len(vecs) == len(means) == len(weights)
        zero = lambda: torch.zeros(1, dtype=torch.float, device=device).squeeze()
        loss = zero()
        for vec, mean, weight in zip(vecs, means, weights):
            margin = torch.norm(vec - mean, p=1, dim=1) - self.delta_v
            loss += weight * torch.mean(torch.max(margin, zero()) ** 2)
        loss /= max(1.0, len(vecs))
        return loss

    def compute_loss_ext(
        self,
        means: list[torch.Tensor],
        weights: list[float],
        mext: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute the external term of the loss."""
        assert len(means) == len(weights)
        zero = lambda: torch.zeros(1, dtype=torch.float, device=device).squeeze()
        loss = zero()
        count = len(means)
        if (count > 1) and (mext is not None):
            means0 = torch.stack(means)
            means1 = means0.unsqueeze(0)  # 1 x N x Dim
            means2 = means0.unsqueeze(1)  # N x 1 x Dim
            margin = 2 * self.delta_d - torch.norm(means2 - means1, p=1, dim=2)
            margin = margin[mext.to(device)]
            loss = torch.sum(torch.max(margin, zero()) ** 2)
            loss /= max(1.0, count * (count - 1.0))  # Normalize
        return loss

    def compute_loss_nrm(self, means: list[torch.Tensor], device: torch.device) -> torch.Tensor:
        """Compute the regularization term of the loss."""
        zero = lambda: torch.zeros(1, dtype=torch.float, device=device).squeeze()
        loss = zero()
        if len(means) > 0:
            loss = torch.mean(torch.norm(torch.stack(means), p=1, dim=1))
        return loss

    def generate_vecs(
        self,
        embd: torch.Tensor,
        trgt: torch.Tensor,
        mask: torch.Tensor,
        ids: Sequence[int],
    ) -> list[torch.Tensor]:
        """
        Generate a list of vectorized embeddings for each ground truth object.
        """
        if self.mask_background and 0 in ids:
            raise ValueError("ID '0' is not allowed when mask_background is enabled.")

        mask_bool = mask.bool() if not self.mask_background else None
        result = []

        for obj_id in ids:
            obj_mask = (trgt == int(obj_id)) & mask_bool if mask_bool is not None else (trgt == int(obj_id))
            idx = torch.nonzero(obj_mask, as_tuple=True)

            if idx[0].numel() == 0:
                # If there are no indices for this ID, skip to the next one
                continue

            vec = embd[0, :, idx[-3], idx[-2], idx[-1]].transpose(0, 1)  # Count x Dim
            result.append(vec)

        return result

    def compute_ext_matrix(
        self,
        ids: Sequence[int],
        groups: Sequence[Sequence[int]] | None = None,
        recompute_ext: bool = False,
        device: torch.device | None = None,
    ) -> torch.Tensor | None:
        """
        Compute a matrix that indicates the presence of 'external' interaction
        between objects.
        """
        num_ids = len(ids)

        # Recompute external matrix
        if recompute_ext:
            assert groups is not None
            mext_np = np.ones((num_ids, num_ids)) - np.eye(num_ids)
            idmap = {x: i for i, x in enumerate(ids)}
            for group in groups:
                for i, id_i in enumerate(group):
                    for id_j in group[i + 1 :]:
                        mext_np[idmap[id_i], idmap[id_j]] = 0
                        mext_np[idmap[id_j], idmap[id_i]] = 0
            mext = torch.from_numpy(mext_np).to(device, dtype=torch.bool)
        else:
            mext = ~torch.eye(num_ids, dtype=torch.bool, device=device)

        # Safeguard
        if mext.sum() == 0:
            return None

        return mext
