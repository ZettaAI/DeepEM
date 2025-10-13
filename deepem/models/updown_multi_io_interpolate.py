import torch
import torch.nn as nn
import torch.nn.functional as F

from emvision.models import rsunet_act_in, rsunet_act_gn
from deepem.models.layers import Conv, Crop, Scale


def create_model(opt):
    width = opt.width if opt.width else [16, 32, 64, 128, 256, 512]
    depth = len(width) if opt.width else opt.depth
    if opt.group > 0:
        # Group normalization
        core = rsunet_act_gn(width=width[:depth], group=opt.group, eps=opt.group_eps, act=opt.act)
    else:
        # Batch normalization
        core = rsunet_act_in(width=width[:depth], act=opt.act)
    return Model(core, opt.in_spec, opt.out_spec, width[0], crop=opt.crop, onnx=opt.onnx,
                scale_init=opt.scale_init, scale_factor=opt.updown_scale_factor)


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


class DownBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        # Handle dictionary input
        if isinstance(x, dict):
            return {k: F.interpolate(v, size=self.size, mode='trilinear', align_corners=False)
                   for k, v in x.items()}

        # Handle single tensor input
        if torch.is_tensor(x):
            return F.interpolate(x, size=self.size, mode='trilinear', align_corners=False)

        # Handle tuple of tensors
        return tuple(F.interpolate(xi, size=self.size, mode='trilinear', align_corners=False)
                    for xi in x)


class UpBlock(nn.Module):
    def __init__(self, out_spec, size):
        super().__init__()
        self.blocks = nn.ModuleDict({
            k: nn.Upsample(
                size=size,
                mode='trilinear',
                recompute_scale_factor=False
            ) for k in out_spec
        })

    def forward(self, x):
        # Handle dictionary input
        if isinstance(x, dict):
            return {k: block(x[k]) for k, block in self.blocks.items()}

        # Handle tuple input
        assert len(x) == len(self.blocks), f"Expected {len(self.blocks)} inputs, got {len(x)}"
        return {k: block(xi) for (k, block), xi in zip(self.blocks.items(), x)}


class Model(nn.Sequential):
    """
    Residual Symmetric U-Net with down/upsampling for multiple inputs/outputs.
    """
    def __init__(self, core, in_spec, out_spec, out_channels, io_kernel=(1, 5, 5),
                crop=None, scale_init=1.0, scale_factor=(1, 2, 2), onnx=False):
        super().__init__()

        in_size = next(iter(in_spec.values()))[-3:]
        assert all(spec[-3:] == in_size for spec in in_spec.values()), "All input sizes must be equal"
        assert all(s % f == 0 for s, f in zip(in_size, scale_factor)), "Input sizes must be divisible by scale factors"
            
        # Convert to integers using integer division
        new_size = tuple(int(s / f) for s, f in zip(in_size, scale_factor))

        self.add_module('down', DownBlock(size=new_size))
        self.add_module('in', InputBlock(in_spec, out_channels, io_kernel))
        self.add_module('core', core)
        self.add_module('out', OutputBlock(out_channels, out_spec, io_kernel, onnx=onnx, scale_init=scale_init))
        self.add_module('up', UpBlock(out_spec, size=in_size))
        if crop is not None:
            self.add_module('crop', Crop(crop)) 