# MALIS Integration Summary

This document summarizes the integration of the PyTorch-compatible MALIS library into DeepEM.

## Overview

The MALIS (MAximum LIkelihood Segmentation) loss function has been integrated into DeepEM using the updated PyTorch-compatible `malis` package. This integration provides proper autograd support and efficient computation for training segmentation models.

## Changes Made

### 1. Updated `deepem/loss/malis.py`

**Key improvements:**
- Removed custom numpy-based MALIS implementation
- Now uses `malis.malis_weights_op()` from the PyTorch-compatible malis package
- Proper autograd support for gradient computation
- Cleaner, more maintainable code

**Removed code:**
- `nodelist_like()` - Now uses `malis.nodelist_like()` from the package
- `compute_malis_weights_python()` - Fallback implementation removed (requires package)
- `MalisWeights` class - Replaced by `malis.MalisWeights` and `malis.malis_weights_op()`

**Updated classes:**
- `MALISLoss`: Now uses `malis.malis_weights_op()` for weight computation
  - Supports batched inputs
  - Handles masks properly
  - Maintains compatibility with DeepEM's loss API
- `MALISAffinityLoss`: Integration with DeepEM's `AffinityLoss` framework
  - Unchanged API, works as before
  - Now benefits from PyTorch-native malis implementation

### 2. Updated `deepem/loss/__init__.py`

Already includes the MALIS loss exports:
```python
from .malis import MALISLoss, MALISAffinityLoss
```

## Dependencies

### Required
- PyTorch >= 1.9.0
- malis package (PyTorch-compatible version)

### Installation

Install the PyTorch-compatible malis package:
```bash
# From source (pytorch-conversion branch)
cd ~/Workbench/ZettaAI/malis
pip install -e .

# Or if available on PyPI
pip install malis
```

## API Usage

### Direct MALIS Loss

```python
from deepem.loss import MALISLoss

# Define edges (3D nearest neighbors)
edges = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]

# Create loss function
malis_loss = MALISLoss(edges=edges, size_average=True)

# Forward pass
# pred_affs: (num_edges, z, y, x) - predicted affinities
# gt_affs: (num_edges, z, y, x) - ground truth affinities
# gt_seg: (z, y, x) - ground truth segmentation
# mask: (z, y, x) - optional mask
loss, nmsk = malis_loss(pred_affs, gt_affs, gt_seg, mask)

# Backward pass
loss.backward()
```

### AffinityLoss Framework Integration

```python
from deepem.loss import MALISAffinityLoss

# Create loss function
malis_loss = MALISAffinityLoss(
    edges=edges,
    split_boundary=True,
    size_average=False
)

# Forward pass (same API as other AffinityLoss variants)
# preds: (batch, num_edges, z, y, x) - predicted affinities
# label: (batch, z, y, x) - ground truth segmentation
# mask: (batch, z, y, x) - optional mask
loss, nmsk = malis_loss(preds, label, mask)
```

## Key Features

1. **PyTorch Native**: Full autograd support for gradient computation
2. **Efficient**: Uses optimized C++ backend from malis package
3. **Batched Operations**: Supports batched inputs for training
4. **Masking Support**: Handles masks for partial labels and boundaries
5. **Compatible**: Drop-in replacement in existing DeepEM training pipelines

## Testing

Run the test suite to verify the integration:
```bash
cd ~/Workbench/ZettaAI/DeepEM
python test_malis_loss.py
```

The test suite includes:
- Basic MALIS loss computation
- Batched inputs
- Different spatial dimensions
- Perfect prediction (should give near-zero loss)
- Gradient computation verification

## Architecture

```
User Code (DeepEM training)
         ↓
MALISLoss / MALISAffinityLoss
         ↓
malis.malis_weights_op()
         ↓
MalisWeightsFunction.apply() (PyTorch autograd)
         ↓
malis.MalisWeights.get_edge_weights()
         ↓
[Tensor → NumPy conversion]
         ↓
C++ Backend (malis_loss_weights)
         ↓
[NumPy → Tensor conversion]
         ↓
PyTorch Tensor (with gradients)
```

## Migration Notes

For existing DeepEM users:

1. **No API changes**: The `MALISLoss` and `MALISAffinityLoss` classes maintain the same API
2. **Install malis**: Ensure the PyTorch-compatible malis package is installed
3. **No code changes needed**: Existing training scripts should work without modification

## Performance Considerations

- **C++ Backend**: Core MALIS computation uses optimized C++ code
- **Memory**: Temporary NumPy conversion required for C++ backend
- **GPU Support**: Tensors are transferred to CPU for computation, then back to original device
- **Gradients**: Proper gradient flow maintained through PyTorch autograd system

## References

- MALIS Paper: [Turaga et al. 2009](http://papers.nips.cc/paper/3887-maximin-affinity-learning-of-image-segmentation)
- MALIS Library: https://github.com/TuragaLab/malis
- PyTorch Conversion: See `PYTORCH_CONVERSION.md` in malis repository
