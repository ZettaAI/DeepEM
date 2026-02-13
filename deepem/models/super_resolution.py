import torch
import torch.nn as nn

from emvision.models import rsunet_act_in, rsunet_act_gn

from deepem.models.layers import Conv, Crop, ShuffleZ


def create_model(opt):
    if opt.width:
        width = opt.width
        depth = len(width)
    else:
        width = [32, 64, 128, 256, 512]
        depth = opt.depth
    if opt.group > 0:
        # Group normalization
        core = rsunet_act_gn(width=width[:depth], group=opt.group, eps=opt.group_eps, act=opt.act)
    else:
        # Instance normalization (default)
        core = rsunet_act_in(width=width[:depth], act=opt.act)

    # Super-resolution mode
    assert opt.sr_mode
    assert opt.sr_scale_z > 0
    return Model(
        core,
        opt.in_spec,
        opt.out_spec,
        width[0],
        opt.sr_scale_z,
        crop=opt.crop,
    )


class InputBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(InputBlock, self).__init__()
        self.add_module('conv', Conv(in_channels, out_channels, kernel_size))


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_spec, scale_z, kernel_size):
        super(OutputBlock, self).__init__()
        for k, v in out_spec.items():
            out_channels = v[-4] * scale_z
            self.add_module(k, nn.Sequential(
                Conv(in_channels, out_channels, kernel_size, bias=True),
                ShuffleZ(scale_z)
            ))

    def forward(self, x):
        return {k: m(x) for k, m in self.named_children()}


class Model(nn.Sequential):
    """
    Super-resolution model with Z-upsampling via ShuffleZ.

    Takes anisotropic input (e.g., 8x8x40 nm) and produces isotropic output
    (e.g., 8x8x8 nm) by upsampling in Z dimension.
    """
    def __init__(self, core, in_spec, out_spec, out_channels, scale_z,
                 io_kernel=(1, 5, 5), crop=None):
        super(Model, self).__init__()
        assert len(in_spec) == 1, "model takes a single input"
        in_channels = list(in_spec.values())[0][0]
        self.add_module('in', InputBlock(in_channels, out_channels, io_kernel))
        self.add_module('core', core)
        self.add_module('out', OutputBlock(out_channels, out_spec, scale_z, io_kernel))
        if crop is not None:
            self.add_module('crop', Crop(crop))
