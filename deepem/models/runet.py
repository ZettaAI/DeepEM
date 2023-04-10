from __future__ import annotations

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

    if not hasattr(opt, "conv_mode"):
        opt.conv_mode = "valid"

    if not opt.norm:
        core = emvision.models.RUNet(width=width[:depth], norm=None, mode=opt.conv_mode)
    else:
        core = emvision.models.RUNet(width=width[:depth], mode=opt.conv_mode)
    return Model(
        core, opt.in_spec, opt.out_spec, width[0], mode=opt.conv_mode, onnx=opt.onnx
    )


class InputBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, mode="valid"):
        super(InputBlock, self).__init__()
        self.add_module('conv', Conv(in_channels, out_channels, kernel_size, mode=mode))
        self.crop_margin = utils.crop_margin(kernel_size, mode=mode)


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_spec, kernel_size, mode="valid", onnx=False):
        super(OutputBlock, self).__init__()
        self.onnx = onnx
        for k, v in out_spec.items():
            out_channels = v[-4]
            self.add_module(
                k, Conv(in_channels, out_channels, kernel_size, bias=True, mode=mode)
            )
        self.crop_margin = utils.crop_margin(kernel_size, mode=mode)

    def forward(self, x):
        if self.onnx:
            return tuple(m(x) for k, m in self.named_children())
        else:
            return {k: m(x) for k, m in self.named_children()}


class AutoPad(nn.Module):
    def __init__(self, crop_margin: tuple[int, int, int]):
        super(AutoPad, self).__init__()
        self.pad_margin = crop_margin

    def forward(self, x):
        # input: tuple of outputs
        if isinstance(x, tuple):
            return tuple(self.pad(output) for output in x)
        elif isinstance(x, torch.Tensor):
            return self.pad(output)

    def pad(self, x):
        padded_size = (
            x.shape[:-3]
            + tuple(p + p + s for p, s in zip(self.pad_margin, x.shape[-3:]))
        )
        beg = self.pad_margin
        end = tuple(s - p for p, s in zip(self.pad_margin, padded_size[-3:]))

        padded = torch.zeros(padded_size, dtype=x.dtype, device=x.device)
        padded[
            ...,
            beg[-3]:end[-3],
            beg[-2]:end[-2],
            beg[-1]:end[-1],
        ] = x

        return padded


class Model(nn.Sequential):
    """
    Residual U-Net.
    """
    def __init__(
        self,
        core,
        in_spec,
        out_spec,
        out_channels,
        io_kernel=(3,3,3),
        mode="valid",
        onnx=False,
    ):
        super(Model, self).__init__()

        assert len(in_spec)==1, "model takes a single input"
        in_channels = list(in_spec.values())[0][-4]

        self.add_module(
            'inblock', InputBlock(in_channels, out_channels, io_kernel, mode=mode)
        )
        self.add_module('core', core)
        self.add_module(
            'outblock',
            OutputBlock(out_channels, out_spec, io_kernel, mode=mode, onnx=onnx),
        )

        self.crop_margin = utils.sum3(
            utils.sum3(
                self.inblock.crop_margin,
                self.outblock.crop_margin,
            ),
            self.core.crop_margin
        )

        if onnx:
            # The purpose of using ONNX is exporting for chunkflow, but chunkflow
            # only works with networks where the input patch size equals the output
            # patch size.
            self.add_module("autopad", AutoPad(self.crop_margin))
