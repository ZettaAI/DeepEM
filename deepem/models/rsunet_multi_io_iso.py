import torch
import torch.nn as nn

from emvision.models import rsunet_act, rsunet_act_gn

from deepem.models.layers import Conv, Crop, Scale


def create_model(opt):
    width = opt.width if opt.width else [16, 32, 64, 128, 256, 512]
    depth = len(width) if opt.width else opt.depth
    zfactor = [2] * (depth - 1)
    if opt.group > 0:
        # Group normalization
        core = rsunet_act_gn(width=width[:depth], zfactor=zfactor, group=opt.group, act=opt.act)
    else:
        # Batch normalization
        core = rsunet_act(width=width[:depth], zfactor=zfactor, act=opt.act)
    return Model(core, opt.in_spec, opt.out_spec, width[0], crop=opt.crop,
                onnx=opt.onnx, scale_init=opt.scale_init)


class InputBlock(nn.Module):
    def __init__(self, in_spec, out_channels, kernel_size):
        super().__init__()
        self.keys = sorted(in_spec.keys())
        self.blocks = nn.ModuleDict({
            k: Conv(v[-4], out_channels, kernel_size)
            for k, v in in_spec.items()
        })

    def forward(self, x):
        # PyTorch training mode - handle dict input
        if isinstance(x, dict):
            x = tuple(x[k] for k in self.keys)

        # Handle single tensor or tuple input
        if torch.is_tensor(x):
            assert len(self.blocks) == 1, "Single tensor input requires exactly one input channel"
            return self.blocks[self.keys[0]](x)

        # Multiple input case (tuple)
        assert len(x) == len(self.blocks), f"Expected {len(self.blocks)} inputs, got {len(x)}"
        return sum(self.blocks[k](xi) for k, xi in zip(self.keys, x))


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
            print("OutputBlock keys:", list(self.blocks.keys()))
            return tuple(m(x) for m in self.blocks.values())
        return {k: m(x) for k, m in self.blocks.items()}


class Model(nn.Sequential):
    """
    Isotropic Residual Symmetric U-Net.
    """
    def __init__(self, core, in_spec, out_spec, out_channels, io_kernel=(5, 5, 5),
                 crop=None, onnx=False, scale_init=1.0):
        super().__init__()
        self.add_module('in', InputBlock(in_spec, out_channels, io_kernel))
        self.add_module('core', core)
        self.add_module('out',
            OutputBlock(out_channels, out_spec, io_kernel, onnx=onnx, scale_init=scale_init))
        if crop is not None:
            self.add_module('crop', Crop(crop)) 