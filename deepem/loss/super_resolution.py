import torch
import torch.nn as nn
import torch.nn.functional as F


class SuperResolutionLoss(nn.Module):
    """
    Dual-loss wrapper for super-resolution training with mixed iso/aniso data.

    For isotropic samples: computes both iso loss (full resolution) and aniso loss
    (downsampled prediction vs downsampled target).

    For anisotropic samples: computes only aniso loss.

    Args:
        base_criterion: The underlying loss function (e.g., BCELoss, AffinityLoss)
        scale_z: Z downsampling factor for computing aniso loss (default: 5)
        iso_weight: Weight for isotropic loss component (default: 1.0)
        aniso_weight: Weight for anisotropic loss component (default: 1.0)
    """
    def __init__(self, base_criterion, scale_z=5, iso_weight=1.0, aniso_weight=1.0):
        super().__init__()
        self.criterion = base_criterion
        self.scale_z = scale_z
        self.iso_weight = iso_weight
        self.aniso_weight = aniso_weight
        self.downsample = nn.AvgPool3d(kernel_size=(scale_z, 1, 1))

    def forward(self, pred, target_iso, target_aniso, mask_iso, mask_aniso, is_isotropic):
        """
        Compute the super-resolution loss.

        Args:
            pred: Network prediction at isotropic resolution (B, C, Z, Y, X)
            target_iso: Isotropic ground truth (B, C, Z, Y, X) - may be None for aniso samples
            target_aniso: Anisotropic ground truth (B, C, Z/scale_z, Y, X)
            mask_iso: Mask for isotropic loss - may be None for aniso samples
            mask_aniso: Mask for anisotropic loss
            is_isotropic: Boolean indicating if this is an isotropic sample

        Returns:
            tuple: (total_loss, num_valid_voxels)
        """
        # Always compute aniso loss
        pred_aniso = self.downsample(pred)
        loss_aniso, nmsk_aniso = self.criterion(pred_aniso, target_aniso, mask_aniso)

        if is_isotropic:
            # Compute iso loss for isotropic samples
            loss_iso, nmsk_iso = self.criterion(pred, target_iso, mask_iso)
            total_loss = self.iso_weight * loss_iso + self.aniso_weight * loss_aniso
            # Return combined mask count (use iso mask as primary)
            return total_loss, nmsk_iso
        else:
            # Only aniso loss for anisotropic samples
            return self.aniso_weight * loss_aniso, nmsk_aniso


class SuperResolutionAffinityLoss(nn.Module):
    """
    Dual-loss wrapper for affinity-based super-resolution training.

    Similar to SuperResolutionLoss but handles the affinity loss specifics
    where the loss is computed per-edge and then combined.

    Args:
        base_criterion: AffinityLoss instance
        scale_z: Z downsampling factor for computing aniso loss (default: 5)
        iso_weight: Weight for isotropic loss component (default: 1.0)
        aniso_weight: Weight for anisotropic loss component (default: 1.0)
    """
    def __init__(self, base_criterion, scale_z=5, iso_weight=1.0, aniso_weight=1.0):
        super().__init__()
        self.criterion = base_criterion
        self.scale_z = scale_z
        self.iso_weight = iso_weight
        self.aniso_weight = aniso_weight
        self.downsample = nn.AvgPool3d(kernel_size=(scale_z, 1, 1))

    def forward(self, pred, target_iso, target_aniso, mask_iso, mask_aniso, is_isotropic):
        """
        Compute the super-resolution affinity loss.

        Args:
            pred: Network prediction at isotropic resolution (B, C, Z, Y, X)
            target_iso: Isotropic affinity ground truth (B, 3, Z, Y, X)
            target_aniso: Anisotropic affinity ground truth (B, 3, Z/scale_z, Y, X)
            mask_iso: Mask for isotropic loss
            mask_aniso: Mask for anisotropic loss
            is_isotropic: Boolean indicating if this is an isotropic sample

        Returns:
            tuple: (total_loss, num_valid_voxels)
        """
        # Always compute aniso loss
        pred_aniso = self.downsample(pred)
        loss_aniso, nmsk_aniso = self.criterion(pred_aniso, target_aniso, mask_aniso)

        if is_isotropic:
            # Compute iso loss for isotropic samples
            loss_iso, nmsk_iso = self.criterion(pred, target_iso, mask_iso)
            total_loss = self.iso_weight * loss_iso + self.aniso_weight * loss_aniso
            return total_loss, nmsk_iso
        else:
            # Only aniso loss for anisotropic samples
            return self.aniso_weight * loss_aniso, nmsk_aniso


class SuperResolutionMeanLoss(nn.Module):
    """
    Dual-loss wrapper for embedding (MeanLoss) in super-resolution training.

    For embedding loss, we compute:
    - Iso loss: MeanLoss on full-resolution embeddings and segmentation
    - Aniso loss: MeanLoss on downsampled embeddings and segmentation

    The downsampling for embeddings uses trilinear interpolation to preserve
    the embedding space structure, while segmentation uses nearest neighbor.

    Args:
        base_criterion: MeanLoss instance
        scale_z: Z downsampling factor (default: 5)
        iso_weight: Weight for isotropic loss component (default: 1.0)
        aniso_weight: Weight for anisotropic loss component (default: 1.0)
    """
    def __init__(self, base_criterion, scale_z=5, iso_weight=1.0, aniso_weight=1.0):
        super().__init__()
        self.criterion = base_criterion
        self.scale_z = scale_z
        self.iso_weight = iso_weight
        self.aniso_weight = aniso_weight

    def _downsample_embedding(self, embd):
        """Downsample embedding using trilinear interpolation."""
        scale = (1.0 / self.scale_z, 1.0, 1.0)
        return F.interpolate(embd, scale_factor=scale, mode='trilinear', align_corners=False)

    def _downsample_labels(self, labels):
        """Downsample segmentation labels using nearest neighbor."""
        scale = (1.0 / self.scale_z, 1.0, 1.0)
        return F.interpolate(labels.float(), scale_factor=scale, mode='nearest').to(labels.dtype)

    def forward(self, pred, target_iso, target_aniso, mask_iso, mask_aniso, is_isotropic,
                splt_iso=None, splt_aniso=None):
        """
        Compute the super-resolution embedding loss.

        Args:
            pred: Network prediction (embeddings) at isotropic resolution (B, C, Z, Y, X)
            target_iso: Isotropic segmentation labels (B, 1, Z, Y, X)
            target_aniso: Anisotropic segmentation labels (B, 1, Z/scale_z, Y, X)
            mask_iso: Mask for isotropic loss
            mask_aniso: Mask for anisotropic loss
            is_isotropic: Boolean indicating if this is an isotropic sample
            splt_iso: Optional split labels for isotropic (for recompute_ext)
            splt_aniso: Optional split labels for anisotropic

        Returns:
            tuple: (total_loss, num_valid_voxels)
        """
        # Always compute aniso loss
        pred_aniso = self._downsample_embedding(pred)
        loss_aniso, nmsk_aniso = self.criterion(pred_aniso, target_aniso, mask_aniso, splt=splt_aniso)

        if is_isotropic:
            # Compute iso loss for isotropic samples
            loss_iso, nmsk_iso = self.criterion(pred, target_iso, mask_iso, splt=splt_iso)
            total_loss = self.iso_weight * loss_iso + self.aniso_weight * loss_aniso
            return total_loss, nmsk_iso
        else:
            # Only aniso loss for anisotropic samples
            return self.aniso_weight * loss_aniso, nmsk_aniso
