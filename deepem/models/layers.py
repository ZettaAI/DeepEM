import torch
import torch.nn as nn

import emvision
from emvision.models.utils import pad_size

from deepem.utils import torch_utils


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                       bias=False, mode="same"):
        super(Conv, self).__init__()
        padding = pad_size(kernel_size, mode)
        self.conv = nn.Conv3d(in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        return x * self.scale


class Crop(nn.Module):
    def __init__(self, crop):
        super(Crop, self).__init__()
        self.crop = tuple(crop)

    def forward(self, x):
        if self.crop is not None:
            for k, v in x.items():
                cropsz = [int(v.shape[i]*self.crop[i]) for i in [-3,-2,-1]]
                x[k] = torch_utils.crop_center(v, cropsz)
        return x


class ShuffleZ(nn.Module):
    """
    Reference: 
        https://github.com/gap370/pixelshuffle3d/blob/master/pixelshuffle3d.py
    """
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, input):
        s = self.scale
        n, c, z, y, x = input.size()
        assert c % s == 0
        input_view = input.contiguous().view(n, c//s, s, z, y, x)
        output = input_view.permute(0, 1, 3, 2, 4, 5).contiguous()
        return output.view(n, c//s, z*s, y, x)
