# MALIS Loss Integration with DeepEM

## Summary

The MALIS (MAximum LIkelihood Segmentation) loss has been successfully integrated into DeepEM using the PyTorch-compatible malis package. This implementation provides proper autograd support and efficient computation for training segmentation models.

**Note**: This replaces the previous implementation that had a Python fallback. The new implementation requires the PyTorch-compatible malis package for correct functionality.

## Implementation Details

### Files Created/Modified

1. **deepem/loss/malis.py** (UPDATED)
   - Main MALIS loss implementation
   - Contains:
     - `MALISLoss`: PyTorch nn.Module that uses `malis.malis_weights_op()` for weight computation
     - `MALISAffinityLoss`: Wrapper for integration with AffinityLoss framework
   - **Removed**: Old numpy-based implementation and Python fallback
   - **Now uses**: PyTorch-native malis package with proper autograd support

2. **deepem/loss/__init__.py** (MODIFIED)
   - Added exports for `MALISLoss` and `MALISAffinityLoss`

3. **test_malis_loss.py** (CREATED)
   - Comprehensive test suite for MALIS loss
   - Tests basic functionality, batched inputs, and edge cases
   - All tests pass successfully

## Key Features

### MALISLoss Module

```python
from deepem.loss import MALISLoss

# Define edge offsets (3D nearest neighbors)
edges = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]

# Create MALIS loss
malis_loss = MALISLoss(
    edges=edges,
    size_average=True,  # Average loss by number of valid samples
    split_boundary=True  # Treat label 0 as boundary
)

# Compute loss
# pred_affs: (num_edges, z, y, x) or (batch, num_edges, z, y, x)
# gt_affs: (num_edges, z, y, x) or (batch, num_edges, z, y, x)
# gt_seg: (z, y, x) or (batch, z, y, x)
# mask: (z, y, x) or (batch, z, y, x)
loss, nmsk = malis_loss(pred_affs, gt_affs, gt_seg, mask)
```

### How MALIS Works

1. **Two-Pass Algorithm**:
   - **Positive pass**: Encourages correct merges (same segment predictions)
   - **Negative pass**: Encourages correct splits (different segment predictions)

2. **Weight Computation**:
   - Uses Union-Find algorithm to track connected components
   - Weights edges by the number of constrained voxel pairs
   - Normalized weights are applied to base loss (MSE by default)

3. **Constrained Optimization**:
   - Ground truth affinities constrain the merging process
   - Masks allow for sparse or partial ground truth

## Dependencies

### Required
- PyTorch >= 1.9.0
- NumPy
- **malis** package (PyTorch-compatible version)
  - Must be installed from the pytorch-conversion branch or compatible version
  - Provides PyTorch-native operations with autograd support
  - Install from: ~/Workbench/ZettaAI/malis or https://github.com/TuragaLab/malis
  - No fallback available - package is required for MALIS loss to work

## Key Improvements

1. **PyTorch Native**: Uses `malis.malis_weights_op()` with proper autograd support
2. **Efficient**: Leverages optimized C++ backend from malis package
3. **Batch Support**: Handles batched inputs natively
4. **DeepEM Integration**: Follows DeepEM's loss module patterns
5. **Clean Code**: Removed redundant numpy implementation, uses malis package directly

## Testing

Run the test suite:

```bash
python test_malis_loss.py
```

Tests include:
- Basic MALIS loss computation
- Batched inputs
- Different spatial sizes
- Perfect prediction (should give ~0 loss)
- Gradient computation

All tests pass successfully.

## Usage Examples

### Basic Usage

```python
import torch
from deepem.loss import MALISLoss

# Define edges (3D)
edges = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]

# Create synthetic data
shape = (16, 16, 16)
pred_affs = torch.rand(3, *shape).requires_grad_(True)  # Predicted affinities
gt_seg = torch.randint(0, 5, shape)  # Ground truth segmentation
gt_affs = torch.randint(0, 2, (3, *shape)).float()  # Ground truth affinities
mask = torch.ones(shape)  # Valid region mask

# Create and apply loss
loss_fn = MALISLoss(edges=edges, size_average=True)
loss, nmsk = loss_fn(pred_affs, gt_affs, gt_seg, mask)

# Backward pass
loss.backward()
```

### Batched Usage

```python
batch_size = 4
pred_affs = torch.rand(batch_size, 3, 16, 16, 16).requires_grad_(True)
gt_seg = torch.randint(0, 5, (batch_size, 16, 16, 16))
gt_affs = torch.randint(0, 2, (batch_size, 3, 16, 16, 16)).float()
mask = torch.ones(batch_size, 16, 16, 16)

loss, nmsk = loss_fn(pred_affs, gt_affs, gt_seg, mask)
```

## Known Limitations

1. **Requires malis package**: Unlike other losses in DeepEM, MALIS requires the external malis package
   - Must install PyTorch-compatible version
   - No fallback implementation available

2. **CPU-bound weight computation**: Weight computation happens on CPU in NumPy (via C++ backend)
   - Weights transferred to GPU after computation
   - This is a limitation of the malis package architecture
   - May add overhead for very large volumes on GPU

## Installation

To use MALIS loss in DeepEM, install the PyTorch-compatible malis package:

```bash
# From local source
cd ~/Workbench/ZettaAI/malis
pip install -e .

# Or from repository (if available)
pip install git+https://github.com/TuragaLab/malis.git@pytorch-conversion
```

The malis package includes:
- Optimized C++ backend for weight computation
- PyTorch autograd integration
- Support for Python 3.10+

## References

- **Paper**: SC Turaga, KL Briggman, M Helmstaedter, W Denk, HS Seung (2009).
  *Maximin learning of image segmentation*.
  Advances in Neural Information Processing Systems (NIPS) 2009.
  http://papers.nips.cc/paper/3887-maximin-affinity-learning-of-image-segmentation

- **Original Implementation**: https://github.com/TuragaLab/malis

## Future Improvements

1. Implement pure PyTorch Union-Find for GPU acceleration
2. Improve MALISAffinityLoss integration
3. Add more sophisticated edge offset patterns
4. Optimize batch processing
5. Add support for 2D images
