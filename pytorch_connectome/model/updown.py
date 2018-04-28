from collections import OrderedDict
import math

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

import emvision
from emvision.models.utils import pad_size


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                       bias=False):
        super(Conv, self).__init__()
        padding = pad_size(kernel_size, 'same')
        self.conv = nn.Conv3d(in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        nn.init.kaiming_normal(self.conv.weight)
        if bias:
            nn.init.constant(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class CaffeBilinearUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CaffeBilinearUp, self).__init__()
        assert in_channels==out_channels
        self.groups = in_channels
        weight = torch.Tensor(self.groups, 1, 1, 4, 4)
        width = weight.size(-1)
        hight = weight.size(-2)
        assert width==hight
        f = float(math.ceil(width / 2.0))
        c = float(width - 1) / (2.0 * f)
        for w in range(width):
            for h in range(hight):
                weight[...,h,w] = (1 - abs(w/f - c)) * (1 - abs(h/f - c))
        self.register_buffer('weight', weight)

    def forward(self, x):
        return F.conv_transpose3d(x, self.weight,
            stride=(1,2,2), padding=(0,1,1), groups=self.groups
        )


class InputBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(InputBlock, self).__init__()
        self.add_module('down', nn.AvgPool3d((1,2,2)))
        self.add_module('conv', Conv(in_channels, out_channels, kernel_size))


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_spec, kernel_size):
        super(OutputBlock, self).__init__()
        self.norm = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        # Sort outputs by name.
        spec = OrderedDict(sorted(out_spec.items(), key=lambda x: x[0]))
        outs = []
        for k, v in spec.items():
            out_channels = v[-4]
            outs.append(nn.Sequential(
                Conv(in_channels, out_channels, kernel_size, bias=True),
                CaffeBilinearUp(out_channels, out_channels)
            ))
        self.outs = nn.ModuleList(outs)

    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        return [out(x) for out in self.outs]


class RSUNet(nn.Sequential):
    """
    Residual Symmetric U-Net with down/upsampling in/output.
    """
    def __init__(self, in_spec, out_spec, depth, **kwargs):
        super(RSUNet, self).__init__()

        assert len(in_spec)==1, "model takes a single input"
        in_channels = list(in_spec.values())[0][0]

        width = [16,32,64,128,256,512]

        self.add_module('in', InputBlock(in_channels, 16, (1,5,5)))
        self.add_module('core', emvision.models.RSUNet(width=width[:depth]))
        self.add_module('out', OutputBlock(16, out_spec, (1,5,5)))
