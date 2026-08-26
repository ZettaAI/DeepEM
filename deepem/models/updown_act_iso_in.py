# DEPRECATED (isotropic z upsampling is broken).
#
# This model builds its core with emvision's rsunet_act_gn / rsunet_act_in,
# whose BilinearUp derives its frozen kernel from the x/y axes only. With
# zfactor > 1 the z taps are all equal: the up-path does not interpolate
# along z (output slices come out in identical pairs) and its z gain is 2
# instead of 1, inflating the up-path against the skip connection at every
# level.
#
# Kept as-is so existing checkpoints keep reproducing. For new training use
# deepem/models/v2/, which builds on deepem/models/core/rsunet.py.
#
# InstanceNorm variant of updown_act_iso.py. That file's group == 0 branch
# builds rsunet_act (BatchNorm), which does not match rsunet_act_iso.py --
# so an rsunet_act_iso checkpoint cannot transfer into it (BatchNorm adds
# running_mean / running_var / num_batches_tracked keys the checkpoint has
# no counterpart for). This variant uses rsunet_act_in, matching
# rsunet_act_iso.py exactly, so the two share a state_dict: 'down' and 'up'
# hold no parameters, and 'in' / 'core' / 'out' keep the same names.
import torch
import torch.nn as nn

import emvision
from emvision.models import rsunet_act_gn, rsunet_act_in

from deepem.models.layers import Conv, Crop, Scale


def create_model(opt):
    if opt.width:
        width = opt.width
        depth = len(width)
    else:
        width = [16,32,64,128,256,512]
        depth = opt.depth

    # Isotropic up & down sampling
    zfactor = [2] * (depth - 1)

    if opt.group > 0:
        # Group normalization
        core = rsunet_act_gn(
            width=width[:depth],
            zfactor=zfactor,
            group=opt.group,
            act=opt.act,
        )
    else:
        # Instance normalization (matches rsunet_act_iso.py)
        core = rsunet_act_in(
            width=width[:depth],
            zfactor=zfactor,
            act=opt.act,
        )
    return Model(core, opt.in_spec, opt.out_spec, width[0], crop=opt.crop,
                 onnx=opt.onnx, scale_init=opt.scale_init)


class InputBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(InputBlock, self).__init__()
        self.add_module('conv', Conv(in_channels, out_channels, kernel_size))


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_spec, kernel_size, onnx=False, scale_init=1.0):
        super(OutputBlock, self).__init__()
        self.onnx = onnx
        for k, v in out_spec.items():
            out_channels = v[-4]
            if k == 'embedding':
                self.add_module(
                    k,
                    nn.Sequential(
                        Conv(in_channels, out_channels, kernel_size, bias=True),
                        Scale(init_value=scale_init),
                    ),
                )
            else:
                self.add_module(k,
                    Conv(in_channels, out_channels, kernel_size, bias=True))

    def forward(self, x):
        if self.onnx:
            return tuple(m(x) for k, m in self.named_children())
        else:
            return {k: m(x) for k, m in self.named_children()}


class DownBlock(nn.Sequential):
    def __init__(self, scale_factor=(1,2,2)):
        super(DownBlock, self).__init__()
        self.add_module('down', nn.AvgPool3d(scale_factor))


class UpBlock(nn.Module):
    def __init__(self, out_spec, scale_factor=(1,2,2), onnx=False):
        super(UpBlock, self).__init__()
        self.onnx = onnx
        for k in out_spec.keys():
            self.add_module(k,
                    nn.Upsample(scale_factor=scale_factor, mode='trilinear'))

    def forward(self, x):
        if self.onnx:
            return tuple(m(x[i]) for i, (_, m) in enumerate(self.named_children()))
        else:
            return {k: m(x[k]) for k, m in self.named_children()}


class Model(nn.Sequential):
    """
    Residual Symmetric U-Net with down/upsampling in/output.
    """
    def __init__(self, core, in_spec, out_spec, out_channels, io_kernel=(5,5,5),
                 scale_factor=(2,2,2), crop=None, onnx=False, scale_init=1.0):
        super(Model, self).__init__()

        assert len(in_spec)==1, "model takes a single input"
        in_channels = 1

        self.add_module('down', DownBlock(scale_factor=scale_factor))
        self.add_module('in', InputBlock(in_channels, out_channels, io_kernel))
        self.add_module('core', core)
        self.add_module('out', OutputBlock(out_channels, out_spec, io_kernel, onnx=onnx, scale_init=scale_init))
        self.add_module('up', UpBlock(out_spec, scale_factor=scale_factor, onnx=onnx))
        if crop is not None:
            self.add_module('crop', Crop(crop))
