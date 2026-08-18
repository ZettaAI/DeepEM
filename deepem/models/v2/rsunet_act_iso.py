"""Isotropic RSUNet built on DeepEM's own core.

Same architecture as ``deepem/models/rsunet_act_iso.py``, but the core comes
from :mod:`deepem.models.core.rsunet` instead of emvision, so the up-path
actually interpolates along z. See that module for the details.
"""

import torch.nn as nn

from deepem.models.core.rsunet import RSUNet, check_onnx_opset
from deepem.models.layers import Conv, Crop, Scale


def create_model(opt):
    check_onnx_opset(opt)

    if opt.width:
        width = opt.width
        depth = len(width)
    else:
        width = [16,32,64,128,256,512]
        depth = opt.depth

    # Isotropic up & down sampling
    zfactor = [2] * (depth - 1)

    core = RSUNet(
        width=width[:depth],
        zfactor=zfactor,
        norm=getattr(opt, 'norm', 'auto'),
        group=opt.group,
        eps=opt.group_eps,
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


class Model(nn.Sequential):
    """
    Residual Symmetric U-Net.
    """
    def __init__(self, core, in_spec, out_spec, out_channels, io_kernel=(5,5,5),
                 crop=None, onnx=False, scale_init=1.0):
        super(Model, self).__init__()

        assert len(in_spec)==1, "model takes a single input"
        in_channels = 1

        self.add_module('in', InputBlock(in_channels, out_channels, io_kernel))
        self.add_module('core', core)
        self.add_module('out', OutputBlock(out_channels, out_spec, io_kernel, onnx=onnx, scale_init=scale_init))
        if crop is not None:
            self.add_module('crop', Crop(crop))
