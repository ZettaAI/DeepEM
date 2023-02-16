import torch
import torch.nn as nn

import emvision
from emvision.models import rsunet_act, rsunet_act_gn

from deepem.models.layers import Conv, Crop, ShuffleZ


def create_model(opt):
    if opt.width:
        width = opt.width
        depth = len(width)
    else:
        width = [16,32,64,128,256,512]
        depth = opt.depth
    if opt.group > 0:
        # Group normalization
        core = rsunet_act_gn(width=width[:depth], group=opt.group, act=opt.act)
    else:
        # Batch normalization
        core = rsunet_act(width=width[:depth], act=opt.act)

    # Tilt series super-resolution
    assert opt.tilt_series > 0
    assert (opt.tilt_series_in > 0) and (opt.tilt_series_out > 0)
    assert opt.tilt_series_in % opt.tilt_series_out == 0
    scale = opt.tilt_series_in // opt.tilt_series_out
    return Model(core, opt.in_spec, opt.out_spec, 
                 opt.tilt_series, width[0], scale, 
                 crop=opt.crop)


class InputBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(InputBlock, self).__init__()
        self.add_module('conv', Conv(in_channels, out_channels, kernel_size))


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_spec, scale, kernel_size):
        super(OutputBlock, self).__init__()
        for k, v in out_spec.items():
            out_channels = v[-4] * scale
            self.add_module(k, nn.Sequential(
                    Conv(in_channels, out_channels, kernel_size, bias=True),
                    ShuffleZ(scale)
                ))

    def forward(self, x):
        return {k: m(x) for k, m in self.named_children()}


class Model(nn.Sequential):
    """
    Residual Symmetric U-Net.
    """
    def __init__(self, core, in_spec, out_spec, in_channels, out_channels, scale,
                 io_kernel=(1,5,5), crop=None):
        super(Model, self).__init__()
        assert len(in_spec)==1, "model takes a single input"
        self.add_module('in', InputBlock(in_channels, out_channels, io_kernel))
        self.add_module('core', core)
        self.add_module('out', OutputBlock(out_channels, out_spec, scale, io_kernel))
        if crop is not None:
            self.add_module('crop', Crop(crop))
