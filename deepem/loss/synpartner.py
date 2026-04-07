"""
Combined loss for Synapse Detection and Synaptic Partner Assignment.

Three pillars:
    L_anchor : Distance-transform penalty keeping deformable anchors inside masks
    L_embed  : Instance embedding clustering (pull/push) in pre & post spaces
    L_arrow  : Partner arrow regression connecting post-anchor to pre-anchor

Training data contract (sample dict keys):
    emb_pre, emb_post       : not used as targets (self-supervised via masks)
    offset                  : not used as target (learned via anchor + arrow losses)
    v_arrow                 : not used as target directly

    synpartner_pre_points   : (N_syn, 3) float tensor — noisy pre-synaptic coords (z,y,x)
    synpartner_post_points  : (N_syn, 3) float tensor — noisy post-synaptic coords (z,y,x)
    synpartner_pre_mask     : (1, 1, D, H, W) — pre-synaptic neuron instance seg
    synpartner_post_mask    : (1, 1, D, H, W) — post-synaptic neuron instance seg
    synpartner_pre_dt       : (1, 1, D, H, W) — distance transform of pre masks
                              (0 inside, >0 outside; precomputed in data loader)
    synpartner_post_dt      : (1, 1, D, H, W) — distance transform of post masks
    synpartner_mask         : (1, 1, D, H, W) — valid region mask
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Coordinate utilities
# ---------------------------------------------------------------------------

def _normalize_coords(coords: torch.Tensor, spatial_size: tuple[int, ...]) -> torch.Tensor:
    """Convert voxel coordinates to grid_sample's [-1, 1] normalized range.

    Args:
        coords: (..., 3) in voxel space, order (z, y, x).
        spatial_size: (D, H, W) of the volume.

    Returns:
        (..., 3) normalized to [-1, 1], reordered to (x, y, z) for grid_sample.
    """
    size = torch.tensor(spatial_size, dtype=coords.dtype, device=coords.device)
    # Voxel center alignment: v_norm = 2 * v / (S - 1) - 1
    # This matches grid_sample(..., align_corners=True).
    norm = 2.0 * coords / (size - 1).clamp(min=1) - 1.0
    # grid_sample expects (x, y, z) ordering — reverse from (z, y, x).
    return norm.flip(-1)


def _sample_volume(volume: torch.Tensor, coords_zyx: torch.Tensor) -> torch.Tensor:
    """Sample a 5-D volume at sub-voxel (z, y, x) coordinates via grid_sample.

    Args:
        volume: (N, C, D, H, W)
        coords_zyx: (N, K, 3) — K query points per batch item, in voxel (z,y,x).

    Returns:
        (N, C, K) sampled values (trilinearly interpolated).

    Gradient flows back through both `volume` and `coords_zyx`.
    """
    N, C, D, H, W = volume.shape
    K = coords_zyx.shape[1]

    # grid_sample needs a 5-D grid: (N, D_out, H_out, W_out, 3)
    # We treat K points as a 1×1×K "volume" so D_out=1, H_out=1, W_out=K.
    grid = _normalize_coords(coords_zyx, (D, H, W))   # (N, K, 3)  in (x,y,z)
    grid = grid.view(N, 1, 1, K, 3)                    # (N, 1, 1, K, 3)

    sampled = F.grid_sample(
        volume, grid,
        mode='bilinear',       # trilinear in 3-D
        padding_mode='border', # clamp to boundary — safe for near-edge anchors
        align_corners=True,
    )
    # sampled: (N, C, 1, 1, K) -> (N, C, K)
    return sampled.view(N, C, K)


# ---------------------------------------------------------------------------
# Loss components
# ---------------------------------------------------------------------------

class AnchorPenalty(nn.Module):
    """L_anchor: penalize deformable anchors that escape their neuron mask.

    Uses precomputed distance-transform volumes (0 inside mask, Euclidean
    distance to boundary outside).  Sampling via grid_sample gives a smooth,
    differentiable penalty whose gradient naturally pushes the anchor back
    inside the mask.

    The DT should be computed per-neuron-instance in the data loader using
    `scipy.ndimage.distance_transform_edt` on the *inverted* binary mask.
    """
    def forward(
        self,
        dt_volume: torch.Tensor,   # (N, 1, D, H, W)
        anchor_coords: torch.Tensor,  # (N, K, 3) voxel zyx
    ) -> torch.Tensor:
        """Returns scalar mean penalty (0 when all anchors are inside)."""
        # (N, 1, K) -> (N, K)
        dt_at_anchor = _sample_volume(dt_volume, anchor_coords).squeeze(1)
        # Squared penalty for smooth gradients near boundary
        return (dt_at_anchor ** 2).mean()


class EmbeddingClusterLoss(nn.Module):
    """L_embed: discriminative instance embedding loss using deformable anchors.

    Instead of computing the cluster centroid as the mean over all mask voxels
    (as in standard MeanLoss), we use the embedding sampled at the deformable
    anchor E(S) as the centroid.  This creates a direct gradient path from
    the clustering objective through grid_sample back to the offset head.

    Pull: all voxels in mask M -> toward E(S)
    Push: different instance centroids E(S_i) <-> E(S_j) apart
    """
    def __init__(
        self,
        delta_v: float = 0.5,   # pull hinge margin
        delta_d: float = 1.5,   # push hinge margin
        alpha: float = 1.0,     # pull weight
        beta: float = 1.0,      # push weight
        gamma: float = 0.001,   # regularization weight
    ):
        super().__init__()
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(
        self,
        embedding: torch.Tensor,     # (N, c, D, H, W)
        mask_seg: torch.Tensor,       # (N, 1, D, H, W) instance IDs
        valid_mask: torch.Tensor,     # (N, 1, D, H, W) binary
        anchor_coords: torch.Tensor,  # (N, K, 3) one anchor per synapse
        anchor_instance_ids: torch.Tensor,  # (N, K) which instance each anchor belongs to
    ) -> torch.Tensor:
        """Returns scalar loss. Applies independently per batch item."""
        device = embedding.device
        N = embedding.shape[0]
        zero = torch.zeros(1, device=device).squeeze()
        loss = zero.clone()

        for b in range(N):
            emb_b = embedding[b:b+1]       # (1, c, D, H, W)
            seg_b = mask_seg[b, 0]          # (D, H, W)
            vmask_b = valid_mask[b, 0]      # (D, H, W)
            coords_b = anchor_coords[b]     # (K, 3)
            ids_b = anchor_instance_ids[b]  # (K,)

            # Sample centroid embeddings at anchor locations
            # (1, c, K)
            centroids = _sample_volume(emb_b, coords_b.unsqueeze(0)).squeeze(0)  # (c, K)

            unique_ids = ids_b.unique()
            unique_ids = unique_ids[unique_ids != 0]  # skip background
            if len(unique_ids) == 0:
                continue

            means = []
            pull_loss = zero.clone()
            n_instances = 0

            for uid in unique_ids:
                # Find anchors belonging to this instance
                anchor_idx = (ids_b == uid).nonzero(as_tuple=True)[0]
                if len(anchor_idx) == 0:
                    continue

                # Use mean of anchor embeddings as centroid for this instance
                centroid = centroids[:, anchor_idx].mean(dim=1)  # (c,)
                means.append(centroid)

                # Pull: all mask voxels of this instance toward centroid
                inst_mask = (seg_b == uid) & vmask_b.bool()
                idx = torch.nonzero(inst_mask, as_tuple=True)
                if len(idx[0]) == 0:
                    continue

                vecs = emb_b[0, :, idx[0], idx[1], idx[2]].T  # (n_voxels, c)
                dist = torch.norm(vecs - centroid.unsqueeze(0), p=1, dim=1)
                margin = dist - self.delta_v
                pull_loss = pull_loss + torch.mean(torch.clamp(margin, min=0) ** 2)
                n_instances += 1

            if n_instances > 0:
                pull_loss = pull_loss / n_instances

            # Push: different instance centroids apart
            push_loss = zero.clone()
            if len(means) > 1:
                stacked = torch.stack(means)       # (M, c)
                m1 = stacked.unsqueeze(0)           # (1, M, c)
                m2 = stacked.unsqueeze(1)           # (M, 1, c)
                dists = torch.norm(m2 - m1, p=1, dim=2)  # (M, M)
                M = len(means)
                # Upper triangle (exclude diagonal)
                triu_mask = torch.triu(torch.ones(M, M, device=device, dtype=torch.bool), diagonal=1)
                margin = 2 * self.delta_d - dists[triu_mask]
                push_loss = torch.mean(torch.clamp(margin, min=0) ** 2)

            # Regularization: small centroid norms
            reg_loss = zero.clone()
            if means:
                reg_loss = torch.mean(torch.norm(torch.stack(means), p=1, dim=1))

            loss = loss + self.alpha * pull_loss + self.beta * push_loss + self.gamma * reg_loss

        return loss / max(N, 1)


class ArrowRegressionLoss(nn.Module):
    r"""L_arrow: partner connection vector regression.

    The arrow vector sampled at the post-anchor should point to the pre-anchor:
        V_arrow(S_post) ≈ S_pre - S_post

    Uses Smooth-L1 (Huber) loss for robustness to annotation noise, since the
    noisy point coordinates make exact regression targets unreliable.
    """
    def __init__(self, beta: float = 2.0):
        """
        Args:
            beta: Smooth-L1 transition point (in voxels). Targets with error
                  < beta use L2-like behavior; beyond beta, L1-like.
        """
        super().__init__()
        self.beta = beta

    def forward(
        self,
        v_arrow: torch.Tensor,             # (N, 3, D, H, W)
        anchor_pre: torch.Tensor,           # (N, K, 3) voxel zyx
        anchor_post: torch.Tensor,          # (N, K, 3) voxel zyx
    ) -> torch.Tensor:
        """Returns scalar loss."""
        N, _, D, H, W = v_arrow.shape
        K = anchor_post.shape[1]
        if K == 0:
            return torch.zeros(1, device=v_arrow.device).squeeze()

        # Sample V_arrow at post-anchor locations: (N, 3, K)
        v_sampled = _sample_volume(v_arrow, anchor_post)

        # Target: displacement from post-anchor to pre-anchor
        # (N, K, 3) -> (N, 3, K)
        target = (anchor_pre - anchor_post).permute(0, 2, 1)

        return F.smooth_l1_loss(v_sampled, target, beta=self.beta)


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class SynPartnerLoss(nn.Module):
    """Combined loss for the 4-headed synapse partner assignment network.

    Orchestrates the three pillars and handles the coordinate bookkeeping
    (computing deformable anchors from raw annotations + predicted offsets).

    Expected out_spec keys: 'emb_pre', 'emb_post', 'offset', 'v_arrow'
    """
    def __init__(
        self,
        embed_dim: int = 8,
        # Anchor penalty
        w_anchor: float = 1.0,
        # Embedding loss
        w_embed: float = 1.0,
        delta_v: float = 0.5,
        delta_d: float = 1.5,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.001,
        # Arrow loss
        w_arrow: float = 1.0,
        arrow_beta: float = 2.0,
        # Offset clamp (max voxels the offset can shift — prevents divergence)
        max_offset: float = 16.0,
    ):
        super().__init__()
        self.w_anchor = w_anchor
        self.w_embed = w_embed
        self.w_arrow = w_arrow
        self.max_offset = max_offset

        self.anchor_loss = AnchorPenalty()
        self.embed_loss = EmbeddingClusterLoss(
            delta_v=delta_v, delta_d=delta_d,
            alpha=alpha, beta=beta, gamma=gamma,
        )
        self.arrow_loss = ArrowRegressionLoss(beta=arrow_beta)

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        sample: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            preds: Model output dict with keys
                   'emb_pre', 'emb_post', 'offset', 'v_arrow'.
            sample: Training sample dict (see module docstring for keys).

        Returns:
            (loss, nmsk) following DeepEM convention.
        """
        device = preds['offset'].device
        nmsk = torch.tensor([1.0], device=device)

        # Unpack predictions
        emb_pre = preds['emb_pre']      # (N, c, D, H, W)
        emb_post = preds['emb_post']    # (N, c, D, H, W)
        offset = preds['offset']        # (N, 6, D, H, W)
        v_arrow = preds['v_arrow']      # (N, 3, D, H, W)
        N, _, D, H, W = emb_pre.shape

        # Unpack annotations
        P_pre = sample['synpartner_pre_points'].to(device)    # (N, K, 3) zyx
        P_post = sample['synpartner_post_points'].to(device)  # (N, K, 3) zyx
        dt_pre = sample['synpartner_pre_dt'].to(device)       # (N, 1, D, H, W)
        dt_post = sample['synpartner_post_dt'].to(device)     # (N, 1, D, H, W)
        mask_pre = sample['synpartner_pre_mask'].to(device)   # (N, 1, D, H, W)
        mask_post = sample['synpartner_post_mask'].to(device) # (N, 1, D, H, W)
        valid_mask = sample['synpartner_mask'].to(device)     # (N, 1, D, H, W)

        K = P_pre.shape[1]
        if K == 0:
            # No synapses in this patch — graph-connected zero
            loss = (emb_pre.sum() + emb_post.sum() + offset.sum() + v_arrow.sum()) * 0
            return loss, nmsk

        # ---------------------------------------------------------------
        # Compute deformable anchors: S = P + clamp(Offset(P))
        # ---------------------------------------------------------------
        # Sample the 6-channel offset field at the noisy annotation points.
        # offset[:, :3] -> pre offsets (z,y,x), offset[:, 3:] -> post offsets
        offset_at_pre = _sample_volume(offset[:, :3], P_pre)    # (N, 3, K)
        offset_at_post = _sample_volume(offset[:, 3:], P_post)  # (N, 3, K)

        # Clamp offsets to prevent runaway shifts early in training
        offset_at_pre = offset_at_pre.clamp(-self.max_offset, self.max_offset)
        offset_at_post = offset_at_post.clamp(-self.max_offset, self.max_offset)

        # (N, 3, K) -> (N, K, 3)
        S_pre = P_pre + offset_at_pre.permute(0, 2, 1)
        S_post = P_post + offset_at_post.permute(0, 2, 1)

        # Clamp anchors to volume bounds so grid_sample stays in-bounds
        lo = torch.zeros(3, device=device)
        hi = torch.tensor([D - 1, H - 1, W - 1], dtype=torch.float32, device=device)
        S_pre = S_pre.clamp(min=lo, max=hi)
        S_post = S_post.clamp(min=lo, max=hi)

        # ---------------------------------------------------------------
        # Pillar 1: Anchor penalty (keep anchors inside neuron masks)
        # ---------------------------------------------------------------
        L_anchor_pre = self.anchor_loss(dt_pre, S_pre)
        L_anchor_post = self.anchor_loss(dt_post, S_post)
        L_anchor = L_anchor_pre + L_anchor_post

        # ---------------------------------------------------------------
        # Pillar 2: Embedding clustering loss (pre and post independently)
        # ---------------------------------------------------------------
        # Instance IDs at anchor locations (from the GT segmentation)
        # We use the nearest-voxel ID lookup — not differentiable, but only
        # used for grouping (not for gradient).
        pre_ids = self._lookup_ids(mask_pre, S_pre)    # (N, K)
        post_ids = self._lookup_ids(mask_post, S_post)  # (N, K)

        L_embed_pre = self.embed_loss(emb_pre, mask_pre, valid_mask, S_pre, pre_ids)
        L_embed_post = self.embed_loss(emb_post, mask_post, valid_mask, S_post, post_ids)
        L_embed = L_embed_pre + L_embed_post

        # ---------------------------------------------------------------
        # Pillar 3: Arrow regression (post-anchor -> pre-anchor)
        # ---------------------------------------------------------------
        L_arrow = self.arrow_loss(v_arrow, S_pre, S_post)

        # ---------------------------------------------------------------
        # Total
        # ---------------------------------------------------------------
        loss = (self.w_anchor * L_anchor
                + self.w_embed * L_embed
                + self.w_arrow * L_arrow)

        return loss, nmsk

    @staticmethod
    @torch.no_grad()
    def _lookup_ids(
        seg_volume: torch.Tensor,  # (N, 1, D, H, W) integer instance IDs
        coords: torch.Tensor,       # (N, K, 3) float zyx
    ) -> torch.Tensor:
        """Nearest-neighbor instance-ID lookup (no gradient needed)."""
        N, _, D, H, W = seg_volume.shape
        K = coords.shape[1]
        # Round to nearest voxel and clamp
        idx = coords.round().long()
        idx[..., 0].clamp_(0, D - 1)
        idx[..., 1].clamp_(0, H - 1)
        idx[..., 2].clamp_(0, W - 1)
        ids = torch.zeros(N, K, dtype=seg_volume.dtype, device=seg_volume.device)
        for b in range(N):
            ids[b] = seg_volume[b, 0, idx[b, :, 0], idx[b, :, 1], idx[b, :, 2]]
        return ids
