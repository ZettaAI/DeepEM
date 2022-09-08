import torch
import torch.nn as nn

import emvision
from emvision.models import utils

from deepem.models.layers import Conv


def create_model(opt):
    if opt.width:
        width = opt.width
        depth = len(width)
    else:
        width = [16,32,64,128,256,512]
        depth = opt.depth

    if not opt.norm:
        core = emvision.models.RUNet(width=width[:depth], norm=None)
    else:
        core = emvision.models.RUNet(width=width[:depth])
    return Model(core, opt.in_spec, opt.out_spec, width[0])


class InputBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(InputBlock, self).__init__()
        self.add_module('conv', Conv(in_channels, out_channels, kernel_size, mode="valid"))
        self.crop_margin = utils.crop_margin(kernel_size, mode="valid")


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_spec, kernel_size):
        super(OutputBlock, self).__init__()
        for k, v in out_spec.items():
            out_channels = v[-4]
            self.add_module(k,
                    Conv(in_channels, out_channels, kernel_size, bias=True, mode="valid"))
        self.crop_margin = utils.crop_margin(kernel_size, mode="valid")

    def forward(self, x):
        return {k: m(x) for k, m in self.named_children()}


class Model(nn.Sequential):
    """
    Residual U-Net.
    """
    def __init__(self, core, in_spec, out_spec, out_channels, io_kernel=(3,3,3)):
        super(Model, self).__init__()

        assert len(in_spec)==1, "model takes a single input"
        in_channels = 1

        self.add_module('inblock', InputBlock(in_channels, out_channels, io_kernel))
        self.add_module('core', core)
        self.add_module('outblock', OutputBlock(out_channels, out_spec, io_kernel))

        self.crop_margin = utils.sum3(
            utils.sum3(
                self.inblock.crop_margin,
                self.outblock.crop_margin,
            ),
            self.core.crop_margin
        )
