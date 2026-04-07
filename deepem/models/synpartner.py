"""
4-Headed 3D U-Net for Synapse Detection and Synaptic Partner Assignment.

Output heads (per voxel x):
    emb_pre(x)  : c-dim pre-synaptic instance embedding
    emb_post(x) : c-dim post-synaptic instance embedding
    offset(x)   : 6-dim deformable offset (3 for pre + 3 for post)
    v_arrow(x)  : 3-dim connection vector (post-anchor -> pre-anchor)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from emvision.models import rsunet_act_in, rsunet_act_gn
from deepem.models.layers import Conv, Scale


def create_model(opt):
    width = opt.width if opt.width else [16, 32, 64, 128, 256, 512]
    depth = len(width) if opt.width else opt.depth
    if opt.group > 0:
        core = rsunet_act_gn(width=width[:depth], group=opt.group,
                             eps=opt.group_eps, act=opt.act)
    else:
        core = rsunet_act_in(width=width[:depth], act=opt.act)
    return Model(core, opt.in_spec, opt.out_spec, width[0],
                 scale_init=getattr(opt, 'scale_init', 1.0))


class InputBlock(nn.Module):
    def __init__(self, in_spec, out_channels, kernel_size):
        super().__init__()
        self.keys = sorted(in_spec.keys())
        self.blocks = nn.ModuleDict({
            k: Conv(v[-4], out_channels, kernel_size)
            for k, v in in_spec.items()
        })

    def forward(self, x):
        if isinstance(x, dict):
            x = tuple(x[k] for k in self.keys)
        if torch.is_tensor(x):
            return self.blocks[self.keys[0]](x)
        return sum(self.blocks[k](xi) for k, xi in zip(self.keys, x))


class OutputBlock(nn.Module):
    """
    4 parallel output heads from a shared feature map.

    Embedding heads (emb_pre, emb_post) get a learnable Scale layer to
    control initial magnitude, matching the existing DeepEM convention.
    The offset head is initialized with near-zero bias so S ≈ P at the
    start of training (identity initialization).
    """
    EMBEDDING_KEYS = {'emb_pre', 'emb_post'}

    def __init__(self, in_channels, out_spec, kernel_size, scale_init=1.0):
        super().__init__()
        self.blocks = nn.ModuleDict()
        for k, v in out_spec.items():
            out_channels = v[-4]
            if k in self.EMBEDDING_KEYS:
                self.blocks[k] = nn.Sequential(
                    Conv(in_channels, out_channels, kernel_size, bias=True),
                    Scale(init_value=scale_init),
                )
            elif k == 'offset':
                conv = Conv(in_channels, out_channels, kernel_size, bias=True)
                # Identity init: zero weights + small bias -> offset ≈ 0 early on
                nn.init.zeros_(conv.conv.weight)
                nn.init.zeros_(conv.conv.bias)
                self.blocks[k] = conv
            else:
                self.blocks[k] = Conv(in_channels, out_channels, kernel_size,
                                      bias=True)

    def forward(self, x):
        return {k: m(x) for k, m in self.blocks.items()}


class Model(nn.Sequential):
    """
    4-Headed Residual Symmetric U-Net for synaptic partner assignment.

    Architecture:
        input -> InputBlock -> RSUNet core -> OutputBlock -> {
            'emb_pre':  (N, c, D, H, W)  pre-synaptic embedding
            'emb_post': (N, c, D, H, W)  post-synaptic embedding
            'offset':   (N, 6, D, H, W)  deformable offsets [pre_zyx, post_zyx]
            'v_arrow':  (N, 3, D, H, W)  connection vector field
        }
    """
    def __init__(self, core, in_spec, out_spec, out_channels,
                 io_kernel=(1, 5, 5), scale_init=1.0):
        super().__init__()
        self.add_module('in', InputBlock(in_spec, out_channels, io_kernel))
        self.add_module('core', core)
        self.add_module('out', OutputBlock(out_channels, out_spec, io_kernel,
                                          scale_init=scale_init))
