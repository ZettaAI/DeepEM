import torch.nn as nn
import torch.nn.functional as F

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
                scale_init=opt.scale_init, scale_factor=opt.updown_scale_factor)


class InputBlock(nn.Module):
    def __init__(self, in_spec, out_channels, kernel_size):
        super().__init__()
        total_in_channels = sum(v[-4] for v in in_spec.values())
        self.block = Conv(total_in_channels, out_channels, kernel_size)

    def forward(self, x):
        return self.block(x)


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


class DownBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return F.interpolate(x, size=self.size, mode='trilinear', align_corners=False)


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
        return {k: block(x[k]) for k, block in self.blocks.items()}


class Model(nn.Sequential):
    """
    Residual Symmetric U-Net with down/upsampling for multiple inputs/outputs.
    """
    def __init__(self, core, in_spec, out_spec, out_channels, io_kernel=(5, 5, 5),
                crop=None, scale_init=1.0, scale_factor=(1, 2, 2)):
        super().__init__()

        in_size = next(iter(in_spec.values()))[-3:]
        assert all(spec[-3:] == in_size for spec in in_spec.values()), "All input sizes must be equal"
        assert all(s % f == 0 for s, f in zip(in_size, scale_factor)), "Input sizes must be divisible by scale factors"
            
        # Convert to integers using integer division
        new_size = tuple(int(s / f) for s, f in zip(in_size, scale_factor))

        self.add_module('down', DownBlock(size=new_size))
        self.add_module('in', InputBlock(in_spec, out_channels, io_kernel))
        self.add_module('core', core)
        self.add_module('out', OutputBlock(out_channels, out_spec, io_kernel, scale_init=scale_init))
        self.add_module('up', UpBlock(out_spec, size=in_size))
        if crop is not None:
            self.add_module('crop', Crop(crop)) 