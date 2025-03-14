import torch.nn as nn

from emvision.models import rsunet_act, rsunet_act_gn

from deepem.models.layers import Conv, Crop, Scale


def create_model(opt):
    width = opt.width if opt.width else [16, 32, 64, 128, 256, 512]
    depth = len(width) if opt.width else opt.depth
    if opt.group > 0:
        # Group normalization
        core = rsunet_act_gn(width=width[:depth], group=opt.group, act=opt.act)
    else:
        # Batch normalization
        core = rsunet_act(width=width[:depth], act=opt.act)
    return Model(core, opt.in_spec, opt.out_spec, width[0], crop=opt.crop,
                 onnx=opt.onnx, scale_init=opt.scale_init)


class InputBlock(nn.Module):
    def __init__(self, in_spec, out_channels, kernel_size, onnx=False):
        super().__init__()
        self.onnx = onnx
        self.keys = sorted(in_spec.keys())  # Store keys for ONNX mode
        self.blocks = nn.ModuleDict({
            k: Conv(v[-4], out_channels, kernel_size)
            for k, v in in_spec.items()
        })

    def forward(self, x):
        if self.onnx:
            # For ONNX, expect x to be a tuple of tensors in same order as self.keys
            return sum(self.blocks[k](xi) for k, xi in zip(self.keys, x))
        # Normal PyTorch mode - x is a dict
        return sum(m(x[k]) for k, m in self.blocks.items())


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_spec, kernel_size, onnx=False, scale_init=1.0):
        super().__init__()
        self.onnx = onnx
        self.blocks = nn.ModuleDict({
            k: (nn.Sequential(
                Conv(in_channels, v[-4], kernel_size, bias=True),
                Scale(init_value=scale_init))
                if k == 'embedding' else
                Conv(in_channels, v[-4], kernel_size, bias=True))
            for k, v in out_spec.items()
        })

    def forward(self, x):
        if self.onnx:
            return tuple(m(x) for m in self.blocks.values())
        return {k: m(x) for k, m in self.blocks.items()}


class Model(nn.Sequential):
    """
    Residual Symmetric U-Net.
    """
    def __init__(self, core, in_spec, out_spec, out_channels, io_kernel=(1, 5, 5),
                 crop=None, onnx=False, scale_init=1.0):
        super().__init__()
        self.add_module('in', InputBlock(in_spec, out_channels, io_kernel, onnx))
        self.add_module('core', core)
        self.add_module('out',
            OutputBlock(out_channels, out_spec, io_kernel, onnx=onnx, scale_init=scale_init))
        if crop is not None:
            self.add_module('crop', Crop(crop))
