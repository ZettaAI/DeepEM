#!/usr/bin/env python
"""
Example usage of MALIS loss in DeepEM.

This script demonstrates how to use the MALIS loss for training
segmentation networks with affinity predictions.
"""

import numpy as np
import torch
import torch.nn as nn
from deepem.loss import MALISLoss


def create_toy_segmentation(shape=(32, 32, 32), num_objects=5):
    """
    Create a toy segmentation with random objects.

    Args:
        shape: Volume shape (z, y, x)
        num_objects: Number of objects to create

    Returns:
        Segmentation array with labeled objects
    """
    seg = np.zeros(shape, dtype=np.int64)

    # Create random box objects
    for obj_id in range(1, num_objects + 1):
        # Random box size
        box_size = np.random.randint(5, 15, size=3)
        # Random position
        pos = [np.random.randint(0, max(1, shape[i] - box_size[i])) for i in range(3)]

        # Fill box
        seg[
            pos[0]:pos[0] + box_size[0],
            pos[1]:pos[1] + box_size[1],
            pos[2]:pos[2] + box_size[2]
        ] = obj_id

    return seg


def compute_gt_affinities(seg, edges):
    """
    Compute ground truth affinities from segmentation.

    Args:
        seg: Segmentation array (z, y, x)
        edges: List of edge offsets

    Returns:
        Ground truth affinities (num_edges, z, y, x)
    """
    shape = seg.shape
    affs = []

    for dz, dy, dx in edges:
        aff = np.zeros(shape, dtype=np.float32)

        # Valid region for this edge
        z_slice = slice(max(0, -dz), min(shape[0], shape[0] - dz))
        y_slice = slice(max(0, -dy), min(shape[1], shape[1] - dy))
        x_slice = slice(max(0, -dx), min(shape[2], shape[2] - dx))

        z_slice_offset = slice(max(0, dz), min(shape[0], shape[0] + dz))
        y_slice_offset = slice(max(0, dy), min(shape[1], shape[1] + dy))
        x_slice_offset = slice(max(0, dx), min(shape[2], shape[2] + dx))

        # Compute affinity
        seg1 = seg[z_slice, y_slice, x_slice]
        seg2 = seg[z_slice_offset, y_slice_offset, x_slice_offset]

        # 1 if same object (and not background), 0 otherwise
        aff_region = ((seg1 == seg2) & (seg1 > 0) & (seg2 > 0)).astype(np.float32)
        aff[z_slice, y_slice, x_slice] = aff_region

        affs.append(aff)

    return np.stack(affs, axis=0)


class SimpleAffinityNet(nn.Module):
    """Simple 3D CNN for affinity prediction."""

    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv3d(32, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


def main():
    """Demonstrate MALIS loss usage."""
    print("=" * 80)
    print("MALIS Loss Example")
    print("=" * 80)

    # Configuration
    shape = (32, 32, 32)
    edges = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]  # 3D nearest neighbors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create toy data
    print("\nCreating toy segmentation data...")
    gt_seg = create_toy_segmentation(shape, num_objects=3)

    # Create fake image (random noise)
    image = np.random.randn(1, *shape).astype(np.float32)

    # Convert to torch tensors
    image_t = torch.from_numpy(image).to(device)
    gt_seg_t = torch.from_numpy(gt_seg).to(device)
    mask_t = torch.ones(shape, device=device)

    print(f"  Image shape: {image_t.shape}")
    print(f"  GT segmentation shape: {gt_seg_t.shape}")
    print(f"  Number of objects: {len(np.unique(gt_seg)) - 1}")

    # Create model
    print("\nCreating simple affinity prediction network...")
    model = SimpleAffinityNet(in_channels=1, out_channels=len(edges)).to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create MALIS loss
    print("\nCreating MALIS loss...")
    criterion = MALISLoss(edges=edges, size_average=True, split_boundary=True)
    print(f"  Edges: {edges}")
    print(f"  Size average: True")
    print(f"  Split boundary: True")

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    print("\nTraining for a few iterations...")
    num_iterations = 10
    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # Forward pass
        pred_affs_logits = model(image_t)

        # Compute MALIS loss (logits=True by default, so sigmoid is applied internally)
        loss, nmsk = criterion(pred_affs_logits, gt_seg_t, mask_t)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Print progress
        if (iteration + 1) % 2 == 0:
            print(f"  Iteration {iteration + 1:2d}/{num_iterations}: "
                  f"Loss = {loss.item():.6f}, "
                  f"Nmsk = {nmsk.item()}")

    print("\nTraining complete!")

    # Test prediction
    print("\nTesting prediction...")
    model.eval()
    with torch.no_grad():
        pred_affs_logits = model(image_t)
        test_loss, test_nmsk = criterion(pred_affs_logits, gt_seg_t, mask_t)
        pred_affs = torch.sigmoid(pred_affs_logits)

    print(f"  Test loss: {test_loss.item():.6f}")
    print(f"  Prediction shape: {pred_affs.shape}")
    print(f"  Prediction range: [{pred_affs.min().item():.3f}, {pred_affs.max().item():.3f}]")

    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)

    # Additional info
    print("\nNotes:")
    print("  - This is a toy example with random data")
    print("  - For real training, use proper image data and segmentation labels")
    print("  - The MALIS loss works best with the C++ malis package installed")
    print("  - Without it, a fallback implementation is used (less accurate)")


if __name__ == "__main__":
    main()
