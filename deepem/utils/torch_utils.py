import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from deepem.utils import py_utils


def revert_sync_batchnorm(module):
    # this is very similar to the function that it is trying to revert:
    # https://github.com/pytorch/pytorch/blob/c8b3686a3e4ba63dc59e5dcfe5db3430df256833/torch/nn/modules/batchnorm.py#L679
    module_output = module
    if isinstance(module, nn.modules.batchnorm.SyncBatchNorm):
        module_output = nn.BatchNorm3d(module.num_features,
                                               module.eps, module.momentum,
                                               module.affine,
                                               module.track_running_stats)
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child))
    del module
    return module_output


def get_pair_first(arr, edge):
    shape = arr.size()[-3:]
    edge = np.array(edge)
    os1 = np.maximum(edge, 0)
    os2 = np.maximum(-edge, 0)
    ret = arr[..., os1[0]:shape[0]-os2[0],
                   os1[1]:shape[1]-os2[1],
                   os1[2]:shape[2]-os2[2]]
    return ret


def get_pair(arr, edge):
    shape = arr.size()[-3:]
    edge = np.array(edge)
    os1 = np.maximum(edge, 0)
    os2 = np.maximum(-edge, 0)
    arr1 = arr[..., os1[0]:shape[0]-os2[0],
                    os1[1]:shape[1]-os2[1],
                    os1[2]:shape[2]-os2[2]]
    arr2 = arr[..., os2[0]:shape[0]-os1[0],
                    os2[1]:shape[1]-os1[1],
                    os2[2]:shape[2]-os1[2]]
    return arr1, arr2


def crop_border(v, size):
    assert all([a > b for a, b in zip(v.shape[-3:], size[-3:])])
    sz, sy, sx = [s // 2 for s in size[-3:]]
    return v[..., sz:-sz, sy:-sy, sx:-sx]


def crop_center(v, size):
    assert all([a >= b for a, b in zip(v.shape[-3:], size[-3:])])
    z, y, x = size[-3:]
    sx = (v.shape[-1] - x) // 2
    sy = (v.shape[-2] - y) // 2
    sz = (v.shape[-3] - z) // 2
    return v[..., sz:sz+z, sy:sy+y, sx:sx+x]


def crop_center_no_strict(v, size):
    assert v.ndim > 3
    assert len(size) == 3
    idx = [slice(None)] * (v.ndim - 3)
    for x, y in zip(v.shape[-3:], size[-3:]):
        if x > y:
            s = (x - y)//2
            idx.append(slice(s, s + y))
        else:
            idx.append(slice(None))
    return v[idx]


def vec2pca(v):
    assert v.ndimension() == 5
    vec = v.detach().cpu().numpy()
    pca = py_utils.fit_pca(vec)
    vec = py_utils.pca_scale_vec(vec, pca)
    return torch.from_numpy(vec)


def pad_center(
    tensor: torch.Tensor,
    target_size: tuple[int, ...],
) -> torch.Tensor:
    """
    Pad a tensor to match the target size, centering the original content.

    Args:
        tensor: Input tensor to pad
        target_size: Desired output size for the last dimensions

    Returns:
        Padded tensor with dimensions matching target_size
    """
    # Convert inputs to lists for easier manipulation
    current_size = list(tensor.shape[-len(target_size):])
    target_size = list(target_size)

    # Calculate padding
    pad = []
    for c, t in zip(reversed(current_size), reversed(target_size)):
        diff = t - c
        # Handle both positive (need padding) and negative (no padding needed) differences
        pad_before = diff // 2
        pad_after = diff - pad_before  # handles odd-sized differences
        pad.extend([max(0, pad_before), max(0, pad_after)])

    # Reverse pad list since F.pad expects dimensions in reverse order
    return torch.nn.functional.pad(tensor, pad)
