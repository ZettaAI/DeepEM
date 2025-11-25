"""
MALIS (Maximin Affinity Learning of Image Segmentation) loss implementation.

Integrates the PyTorch-compatible MALIS loss from https://github.com/TuragaLab/malis
"""

import numpy as np
import warnings

import torch
import torch.nn as nn


# Try to import the malis package, but don't fail if it's not available
try:
    import malis
    HAS_MALIS = True
except ImportError:
    HAS_MALIS = False
    warnings.warn(
        "malis package not found. MALIS loss will not work correctly. "
        "Install the PyTorch-compatible malis package from: "
        "https://github.com/TuragaLab/malis"
    )


class MALISLoss(nn.Module):
    """
    MALIS (Maximin Affinity Learning of Image Segmentation) loss.

    This loss function uses the maximin principle to weight edge-wise losses
    based on the segmentation structure. It takes:
    - Predicted affinities
    - Ground truth segmentation
    - Optional mask

    The loss is computed as a weighted sum of edge-wise squared errors,
    where the weights are determined by the MALIS algorithm.

    This implementation uses the malis package functions:
    - malis.seg_to_affgraph: Convert segmentation to affinities
    - malis.segmask_to_affmask: Convert segmentation mask to affinity mask
    - malis.malis_weights_op: Compute MALIS weights with autograd support
    """

    def __init__(self, edges, size_average=True, split_boundary=True,
                 class_balancer=None, logits=True, **kwargs):
        """
        Args:
            edges: List of edge offsets, e.g., [(0,0,1), (0,1,0), (1,0,0)]
            size_average: If True, average the loss by the number of valid samples
            split_boundary: If True, treat boundary (label 0) separately.
                          Note: malis package currently only supports split_boundary=True
            class_balancer: Optional class balancer (not used by MALIS, kept for compatibility)
            logits: If True, apply sigmoid to pred_affs before computing loss
        """
        super(MALISLoss, self).__init__()
        self.size_average = size_average
        self.split_boundary = split_boundary
        self.class_balancer = class_balancer
        self.logits = logits

        # Warn if split_boundary=False since malis package doesn't support it
        if not split_boundary:
            warnings.warn(
                "MALIS loss with split_boundary=False is not currently supported by the malis package. "
                "The boundary (label 0) will be treated as background. "
                "Set split_boundary=True to suppress this warning.",
                UserWarning
            )

        # Convert edges to numpy array for malis functions
        # MALIS package has a different edge convention, so invert the signs
        inverted_edges = [tuple(-e for e in edge) for edge in edges]
        self.neighborhood = np.array(inverted_edges, dtype=np.int32)

    def forward(self, preds, label, mask):
        """
        Compute MALIS loss.

        Args:
            preds: Predicted affinities, shape (batch, num_edges, z, y, x) or (num_edges, z, y, x)
            label: Ground truth segmentation, shape (batch, z, y, x) or (z, y, x)
            mask: Mask, shape (batch, z, y, x) or (z, y, x)

        Returns:
            loss: Scalar loss value
            nmsk: Number of valid samples (for compatibility with other losses)
        """
        if not HAS_MALIS:
            raise RuntimeError(
                "MALIS loss requires the malis package. "
                "Install it from: https://github.com/TuragaLab/malis"
            )

        # Handle batch dimension
        has_batch = preds.dim() == 5
        if has_batch:
            batch_size = preds.shape[0]
            losses = []
            nmasks = []

            for b in range(batch_size):
                loss_b, nmsk_b = self.forward(
                    preds[b],
                    label[b] if label.dim() == 4 else label,
                    mask[b] if mask.dim() == 4 else mask
                )
                losses.append(loss_b)
                nmasks.append(nmsk_b)

            total_loss = sum(losses)
            total_nmsk = sum(nmasks)

            if self.size_average and total_nmsk.item() > 0:
                total_loss = total_loss / total_nmsk.item()
                total_nmsk = torch.tensor(1, dtype=total_nmsk.dtype, device=total_nmsk.device)

            return total_loss, total_nmsk

        # Extract predicted affinities
        pred_affs = preds  # Shape: (num_edges, z, y, x)

        # Remove channel dimension from label if present
        label_seg = label.squeeze(0) if label.dim() == 4 else label

        # Make sure segmentation has correct dtype
        label_seg = label_seg.long() if label_seg.dtype != torch.int64 else label_seg

        # Use malis.seg_to_affgraph to generate ground truth affinities
        # This ensures compatibility with the malis package
        label_seg_np = label_seg.cpu().numpy()
        gt_affs_np = malis.seg_to_affgraph(label_seg_np, self.neighborhood)

        # Convert back to torch tensor
        gt_affs = torch.from_numpy(gt_affs_np).to(pred_affs.device).float()

        # Use malis.segmask_to_affmask to generate binary affinity mask
        # This ensures compatibility with the malis package
        mask_np = mask.cpu().numpy()
        gt_aff_mask_np = malis.segmask_to_affmask(mask_np, self.neighborhood)

        # Convert back to torch tensor (binary mask for malis)
        gt_aff_mask = torch.from_numpy(gt_aff_mask_np).to(pred_affs.device).float()

        # Create balanced mask by applying class balancer to each edge/channel separately
        if self.class_balancer is not None:
            # Apply class balancer to each edge independently
            balanced_masks = []
            for i in range(gt_aff_mask.shape[0]):
                # Class balancer expects (target, mask)
                edge_balanced_mask = self.class_balancer(gt_affs[i], gt_aff_mask[i])
                balanced_masks.append(edge_balanced_mask)
            balanced_mask = torch.stack(balanced_masks, dim=0)
        else:
            # If no class balancer, use the binary mask
            balanced_mask = gt_aff_mask

        # Apply sigmoid if logits=True
        if self.logits:
            pred_affs = torch.sigmoid(pred_affs)

        # Use malis.malis_weights_op to compute weights with proper autograd support
        weights = malis.malis_weights_op(
            pred_affs,
            gt_affs,
            label_seg,
            self.neighborhood,
            gt_aff_mask=gt_aff_mask,
            gt_seg_unlabelled=None
        )

        # Compute edge-wise squared error
        edge_loss = (pred_affs - gt_affs) ** 2

        # Apply weights
        weighted_loss = weights * edge_loss

        # Apply balanced mask
        weighted_loss = weighted_loss * balanced_mask

        # Sum over all dimensions
        loss = weighted_loss.sum()

        # Count valid samples (non-zero entries in binary mask)
        nmsk = (gt_aff_mask > 0).to(dtype=mask.dtype).sum()

        if nmsk.item() == 0:
            # Return a graph-connected zero
            zero = (pred_affs * balanced_mask).sum() * 0
            return zero, nmsk

        if self.size_average:
            loss = loss / nmsk.item()
            nmsk = torch.tensor(1, dtype=nmsk.dtype, device=nmsk.device)

        return loss, nmsk
