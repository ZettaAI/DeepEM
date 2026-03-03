import torch.nn as nn
import torch.nn.functional as F

from emvision.models import rsunet_act_in, rsunet_act_gn

from deepem.models.layers import Conv, Crop, Scale


def create_model(opt):
    if opt.width:
        width = opt.width
        depth = len(width)
    else:
        width = [16, 32, 64, 128, 256, 512]
        depth = opt.depth
    if opt.group > 0:
        # Group normalization
        core = rsunet_act_gn(width=width[:depth], group=opt.group, act=opt.act)
    else:
        # Instance normalization
        core = rsunet_act_in(width=width[:depth], act=opt.act)
    return Model(core, opt.in_spec, opt.out_spec, width[0], crop=opt.crop,
                 scale_init=opt.scale_init, scale_factor=opt.updown_scale_factor)


class InputBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(InputBlock, self).__init__()
        self.add_module('conv', Conv(in_channels, out_channels, kernel_size))


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_spec, kernel_size, scale_init=1.0):
        super(OutputBlock, self).__init__()
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
        return {k: m(x) for k, m in self.named_children()}


class DownBlock(nn.Module):
    def __init__(self, size):
        super(DownBlock, self).__init__()
        self.size = size

    def forward(self, x):
        return F.interpolate(x, size=self.size, mode='trilinear', align_corners=False)


class UpBlock(nn.Module):
    def __init__(self, out_spec, size):
        super(UpBlock, self).__init__()
        for k, v in out_spec.items():
            self.add_module(k,
                    nn.Upsample(
                        size=size,
                        mode='trilinear',
                        recompute_scale_factor=False,
                    ))

    def forward(self, x):
        return {k: m(x[k]) for k, m in self.named_children()}


class Model(nn.Sequential):
    """
    Residual Symmetric U-Net with down/upsampling in/output.
    """
    def __init__(self, core, in_spec, out_spec, out_channels, io_kernel=(1,5,5),
                 crop=None, scale_init=1.0, scale_factor=(1, 2, 2)):
        super(Model, self).__init__()

        assert len(in_spec)==1, "model takes a single input"
        in_channels = 1
        in_size = in_spec['input'][-3:]
        assert all(s % f == 0 for s, f in zip(in_size, scale_factor))
        new_size = tuple(int(s / f) for s, f in zip(in_size, scale_factor))

        self.add_module('down', DownBlock(size=new_size))
        self.add_module('in', InputBlock(in_channels, out_channels, io_kernel))
        self.add_module('core', core)
        self.add_module('out', OutputBlock(out_channels, out_spec, io_kernel, scale_init=scale_init))
        self.add_module('up', UpBlock(out_spec, size=in_size))
        if crop is not None:
            self.add_module('crop', Crop(crop))
