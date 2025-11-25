#!/usr/bin/env python
"""
Test script to verify MALISLoss uses malis.seg_to_affgraph correctly.
"""

import sys
sys.path.insert(0, '/home/kisuk/Workbench/ZettaAI/malis')

import numpy as np
import torch
from deepem.loss import MALISLoss

def test_malis_affinity_loss():
    """Test MALISLoss with malis.seg_to_affgraph."""
    print("Testing MALISLoss with malis.seg_to_affgraph...")

    # Configuration
    shape = (32, 32, 32)
    edges = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Create simple segmentation with a few objects
    seg = np.zeros(shape, dtype=np.int64)
    seg[8:16, 8:16, 8:16] = 1  # First object
    seg[16:24, 16:24, 16:24] = 2  # Second object

    print(f"  Segmentation shape: {seg.shape}")
    print(f"  Number of objects: {len(np.unique(seg)) - 1}")

    # Create random predictions (requires_grad=True for backward pass)
    pred_affs = torch.randn(len(edges), *shape, device=device, requires_grad=True)

    # Convert segmentation to tensor
    seg_tensor = torch.from_numpy(seg).to(device)

    # Create mask
    mask = torch.ones(shape, device=device)

    # Create MALISLoss
    loss_fn = MALISLoss(
        edges=edges,
        size_average=True,
        split_boundary=True,
        logits=True
    )

    print("  Computing loss...")
    try:
        loss, nmsk = loss_fn(pred_affs, seg_tensor, mask)
        print(f"  ✓ Loss computed successfully: {loss.item():.6f}")
        print(f"  ✓ Nmsk: {nmsk.item()}")

        # Test backward pass
        print("  Testing backward pass...")
        loss.backward()
        print(f"  ✓ Backward pass successful")

        # Verify gradient exists
        if pred_affs.grad is not None:
            print(f"  ✓ Gradients computed (shape: {pred_affs.grad.shape})")
        else:
            print(f"  ✗ No gradients computed")

        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_processing():
    """Test MALISLoss with batched inputs."""
    print("\nTesting MALISLoss with batched inputs...")

    # Configuration
    batch_size = 2
    shape = (16, 16, 16)
    edges = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create segmentation batch
    seg_batch = np.zeros((batch_size, *shape), dtype=np.int64)
    seg_batch[0, 4:12, 4:12, 4:12] = 1
    seg_batch[1, 4:12, 4:12, 4:12] = 2

    # Create random predictions (requires_grad=True for backward pass)
    pred_affs_batch = torch.randn(batch_size, len(edges), *shape, device=device, requires_grad=True)

    # Convert to tensors
    seg_tensor_batch = torch.from_numpy(seg_batch).to(device)
    mask_batch = torch.ones(batch_size, *shape, device=device)

    # Create loss function
    loss_fn = MALISLoss(
        edges=edges,
        size_average=True,
        split_boundary=True,
        logits=True
    )

    print(f"  Batch size: {batch_size}")
    print("  Computing loss...")
    try:
        loss, nmsk = loss_fn(pred_affs_batch, seg_tensor_batch, mask_batch)
        print(f"  ✓ Batch loss computed successfully: {loss.item():.6f}")
        print(f"  ✓ Nmsk: {nmsk.item()}")

        # Test backward pass
        print("  Testing backward pass...")
        loss.backward()
        print(f"  ✓ Backward pass successful")

        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("MALISLoss Test Suite")
    print("=" * 80)

    test1_passed = test_malis_affinity_loss()
    test2_passed = test_batch_processing()

    print("\n" + "=" * 80)
    print("Test Results:")
    print(f"  Single sample test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"  Batch test: {'PASSED' if test2_passed else 'FAILED'}")
    print("=" * 80)

    sys.exit(0 if (test1_passed and test2_passed) else 1)
