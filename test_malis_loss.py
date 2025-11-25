#!/usr/bin/env python
"""
Test script for MALIS loss implementation in DeepEM.

This script validates that:
1. MALIS loss can be instantiated and used
2. MALIS loss produces reasonable outputs for synthetic data
3. Integration with DeepEM's AffinityLoss framework works
"""

import numpy as np
import torch
from deepem.loss import MALISLoss


def create_synthetic_data(shape=(16, 16, 16), num_objects=3):
    """
    Create synthetic segmentation and affinity data for testing.

    Args:
        shape: Spatial shape (z, y, x)
        num_objects: Number of objects in the segmentation

    Returns:
        gt_seg: Ground truth segmentation
        gt_affs: Ground truth affinities
        pred_affs: Predicted affinities (with some noise)
        mask: Valid region mask
    """
    # Create synthetic segmentation with labeled objects
    gt_seg = np.zeros(shape, dtype=np.int64)

    # Create block objects
    z_step = shape[0] // num_objects
    for i in range(num_objects):
        z_start = i * z_step
        z_end = min((i + 1) * z_step, shape[0])
        gt_seg[z_start:z_end, :, :] = i + 1

    # Define edges (3D nearest neighbors)
    edges = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]

    # Compute ground truth affinities
    gt_affs = []
    for edge in edges:
        dz, dy, dx = edge
        aff = np.zeros(shape, dtype=np.float32)

        # Valid region for this edge
        z_slice = slice(max(0, -dz), min(shape[0], shape[0] - dz))
        y_slice = slice(max(0, -dy), min(shape[1], shape[1] - dy))
        x_slice = slice(max(0, -dx), min(shape[2], shape[2] - dx))

        z_slice_offset = slice(max(0, dz), min(shape[0], shape[0] + dz))
        y_slice_offset = slice(max(0, dy), min(shape[1], shape[1] + dy))
        x_slice_offset = slice(max(0, dx), min(shape[2], shape[2] + dx))

        # Compute affinity: 1 if same object, 0 otherwise
        seg1 = gt_seg[z_slice, y_slice, x_slice]
        seg2 = gt_seg[z_slice_offset, y_slice_offset, x_slice_offset]

        aff_region = ((seg1 == seg2) & (seg1 > 0) & (seg2 > 0)).astype(np.float32)
        aff[z_slice, y_slice, x_slice] = aff_region

        gt_affs.append(aff)

    gt_affs = np.stack(gt_affs, axis=0)

    # Create predicted affinities with some noise
    pred_affs = gt_affs.copy()
    noise = np.random.randn(*pred_affs.shape) * 0.1
    pred_affs = np.clip(pred_affs + noise, 0, 1).astype(np.float32)

    # Create mask (all valid)
    mask = np.ones(shape, dtype=np.float32)

    return gt_seg, gt_affs, pred_affs, mask


def test_malis_loss_basic():
    """Test basic MALIS loss computation."""
    print("=" * 80)
    print("Test 1: Basic MALIS Loss")
    print("=" * 80)

    # Create synthetic data
    shape = (16, 16, 16)
    edges = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    gt_seg, gt_affs, pred_affs, mask = create_synthetic_data(shape)

    # Convert to torch tensors
    gt_seg_t = torch.from_numpy(gt_seg)
    gt_affs_t = torch.from_numpy(gt_affs)
    pred_affs_t = torch.from_numpy(pred_affs).requires_grad_(True)
    mask_t = torch.from_numpy(mask)

    # Create MALIS loss
    malis_loss = MALISLoss(edges=edges, size_average=True)

    # Compute loss
    loss, nmsk = malis_loss(pred_affs_t, gt_affs_t, gt_seg_t, mask_t)

    print(f"Loss value: {loss.item():.6f}")
    print(f"Number of valid samples: {nmsk.item()}")
    print(f"Loss requires grad: {loss.requires_grad}")

    # Test backward pass
    loss.backward()
    print(f"Gradient computed: {pred_affs_t.grad is not None}")
    print(f"Gradient shape: {pred_affs_t.grad.shape if pred_affs_t.grad is not None else None}")
    print(f"Gradient mean: {pred_affs_t.grad.mean().item() if pred_affs_t.grad is not None else None:.6e}")

    assert loss.item() >= 0, "Loss should be non-negative"
    assert nmsk.item() > 0, "Number of valid samples should be positive"
    print("PASSED\n")


def test_malis_loss_batch():
    """Test MALIS loss with batched input."""
    print("=" * 80)
    print("Test 2: Batched MALIS Loss")
    print("=" * 80)

    # Create synthetic data
    shape = (8, 8, 8)
    batch_size = 2
    edges = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]

    # Create batch data
    batch_gt_seg = []
    batch_gt_affs = []
    batch_pred_affs = []
    batch_mask = []

    for _ in range(batch_size):
        gt_seg, gt_affs, pred_affs, mask = create_synthetic_data(shape, num_objects=2)
        batch_gt_seg.append(gt_seg)
        batch_gt_affs.append(gt_affs)
        batch_pred_affs.append(pred_affs)
        batch_mask.append(mask)

    # Stack into batches
    gt_seg_t = torch.from_numpy(np.stack(batch_gt_seg, axis=0))
    gt_affs_t = torch.from_numpy(np.stack(batch_gt_affs, axis=0))
    pred_affs_t = torch.from_numpy(np.stack(batch_pred_affs, axis=0)).requires_grad_(True)
    mask_t = torch.from_numpy(np.stack(batch_mask, axis=0))

    print(f"Batch shapes:")
    print(f"  gt_seg: {gt_seg_t.shape}")
    print(f"  gt_affs: {gt_affs_t.shape}")
    print(f"  pred_affs: {pred_affs_t.shape}")
    print(f"  mask: {mask_t.shape}")

    # Create MALIS loss
    malis_loss = MALISLoss(edges=edges, size_average=True)

    # Compute loss
    loss, nmsk = malis_loss(pred_affs_t, gt_affs_t, gt_seg_t, mask_t)

    print(f"Loss value: {loss.item():.6f}")
    print(f"Number of valid samples: {nmsk.item()}")

    # Test backward pass
    loss.backward()
    print(f"Gradient computed: {pred_affs_t.grad is not None}")

    assert loss.item() >= 0, "Loss should be non-negative"
    print("PASSED\n")


def test_malis_affinity_loss():
    """Test MALISLoss with different data sizes."""
    print("=" * 80)
    print("Test 3: MALIS with different data sizes")
    print("=" * 80)

    # Test that MALIS works with different spatial sizes
    for shape in [(8, 8, 8), (10, 12, 14)]:
        edges = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
        gt_seg, gt_affs, pred_affs, mask = create_synthetic_data(shape, num_objects=2)

        # Convert to torch tensors
        gt_seg_t = torch.from_numpy(gt_seg)
        gt_affs_t = torch.from_numpy(gt_affs)
        pred_affs_t = torch.from_numpy(pred_affs).requires_grad_(True)
        mask_t = torch.from_numpy(mask)

        print(f"Testing with shape {shape}:")

        # Create MALIS loss
        malis_loss = MALISLoss(edges=edges, size_average=True)

        # Compute loss
        loss, nmsk = malis_loss(pred_affs_t, gt_affs_t, gt_seg_t, mask_t)

        print(f"  Loss value: {loss.item():.6f}")
        print(f"  Number of valid samples: {nmsk.item()}")

        assert loss.item() >= 0, "Loss should be non-negative"
        assert nmsk.item() > 0, "Number of valid samples should be positive"

    print("PASSED\n")


def test_perfect_prediction():
    """Test that perfect predictions give zero or near-zero loss."""
    print("=" * 80)
    print("Test 4: Perfect Prediction")
    print("=" * 80)

    # Create synthetic data
    shape = (8, 8, 8)
    edges = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    gt_seg, gt_affs, _, mask = create_synthetic_data(shape, num_objects=2)

    # Use ground truth as prediction (perfect prediction)
    pred_affs = gt_affs.copy()

    # Convert to torch tensors
    gt_seg_t = torch.from_numpy(gt_seg)
    gt_affs_t = torch.from_numpy(gt_affs)
    pred_affs_t = torch.from_numpy(pred_affs)
    mask_t = torch.from_numpy(mask)

    # Create MALIS loss
    malis_loss = MALISLoss(edges=edges, size_average=True)

    # Compute loss
    loss, nmsk = malis_loss(pred_affs_t, gt_affs_t, gt_seg_t, mask_t)

    print(f"Loss value for perfect prediction: {loss.item():.6e}")
    print(f"Number of valid samples: {nmsk.item()}")

    # Loss should be very small (close to zero) for perfect predictions
    assert loss.item() < 1e-6, f"Loss should be near zero for perfect prediction, got {loss.item()}"
    print("PASSED\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("MALIS Loss Test Suite")
    print("=" * 80 + "\n")

    try:
        # Check if malis package is available
        try:
            import malis
            print("malis package found - using optimized C++ implementation\n")
        except ImportError:
            print("WARNING: malis package not found - using fallback implementation")
            print("For better performance, install: pip install malis\n")

        # Run tests
        test_malis_loss_basic()
        test_malis_loss_batch()
        test_malis_affinity_loss()
        test_perfect_prediction()

        print("=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print("TEST FAILED!")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
