"""Residual Symmetric U-Net core.

DeepEM's own copy of the RSUNet core, replacing
``emvision.models.rsunet_act_gn`` / ``rsunet_act_in`` for new models.

Differences from the emvision version:

* **Trilinear upsampling.** emvision's ``BilinearUp`` builds a frozen
  Caffe-style transposed-conv kernel from the x/y axes only, broadcasting a
  scalar across the whole z axis of the kernel. With a z factor of 1 the z
  kernel has length 1 and this is harmless, but with a z factor of 2 the four
  z taps are all identical: there is no interpolation along z (output slices
  come out in identical pairs) and the kernel sums to 2 instead of 1 along z,
  inflating the up-path by 2x per level relative to the skip connection.
  ``TrilinearUp`` interpolates on every axis with unit DC gain.
* **Configurable normalization.** One module parameterized by ``norm``
  instead of two near-identical modules, one per norm type.
* **No module-level global state.** emvision stores the activation and the
  group size in module globals, so building two models with different
  settings in one process silently corrupts the first one. Here both are
  constructor arguments.

Module and parameter names deliberately match the emvision version, so a
state dict from an emvision-based model differs only by the ``up.up.0.weight``
buffers that ``BilinearUp`` used to register. Those checkpoints can therefore
be loaded into these models through DeepEM's ``--pretrain`` path.
"""

import torch.nn as nn
import torch.nn.functional as F

from emvision.models.utils import pad_size


__all__ = ['RSUNet', 'TrilinearUp', 'check_onnx_opset']


# ONNX's Resize op only gained `coordinate_transformation_mode` in opset 11.
# Exporting trilinear interpolation to opset 10 succeeds with nothing but a
# warning, and silently produces a graph that does not match PyTorch.
MIN_ONNX_OPSET = 11


def check_onnx_opset(opt):
    """Reject ONNX export at an opset that cannot represent trilinear upsampling."""
    if not getattr(opt, 'onnx', False):
        return
    opset = getattr(opt, 'opset_version', MIN_ONNX_OPSET)
    if opset < MIN_ONNX_OPSET:
        raise ValueError(
            f"--opset_version {opset} cannot represent trilinear upsampling: "
            f"ONNX Resize gained coordinate_transformation_mode in opset "
            f"{MIN_ONNX_OPSET}, and exporting below that silently yields a "
            f"graph whose output does not match PyTorch. "
            f"Use --opset_version {MIN_ONNX_OPSET} or higher."
        )


def conv(in_channels, out_channels, kernel_size=3, stride=1, bias=False):
    padding = pad_size(kernel_size, 'same')
    return nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, bias=bias)


def norm_layer(channels, norm='gn', group=16, eps=1e-5, affine=True):
    """Normalization layer.

    Args:
        norm: one of

            ``'gn'``   group normalization with ``channels // group`` groups
            ``'in'``   instance normalization (per-channel)
            ``'ln'``   layer normalization (all channels at once)
            ``'none'`` no normalization

        group: group *size* for ``'gn'`` -- the number of groups is
            ``channels // group``, matching the emvision convention.
    """
    if norm == 'gn':
        assert group > 0, "group normalization requires --group > 0"
        assert channels % group == 0, \
            f"channels ({channels}) not divisible by group size ({group})"
        return nn.GroupNorm(channels // group, channels, eps=eps, affine=affine)
    if norm == 'in':
        return nn.InstanceNorm3d(channels, eps=eps, affine=affine,
                                 track_running_stats=False)
    if norm == 'ln':
        # LayerNorm over (C,D,H,W) is GroupNorm with a single group. Using
        # GroupNorm keeps the affine parameters channel-shaped and therefore
        # spatial-size agnostic, unlike nn.LayerNorm.
        return nn.GroupNorm(1, channels, eps=eps, affine=affine)
    if norm == 'none':
        return nn.Identity()
    raise ValueError(f"unsupported norm: {norm}")


def act_layer(act='ReLU', act_params=None):
    params = dict(act_params or {})
    assert act in ['ReLU', 'LeakyReLU', 'PReLU', 'ELU'], \
        f"unsupported activation: {act}"
    # Use the in-place module where available.
    if act in ['ReLU', 'LeakyReLU', 'ELU']:
        params['inplace'] = True
    return getattr(nn, act)(**params)


_NORM_KEYS = ('norm', 'group', 'eps', 'affine')
_ACT_KEYS = ('act', 'act_params')


def _norm_kwargs(kwargs):
    return {k: v for k, v in kwargs.items() if k in _NORM_KEYS}


def _act_kwargs(kwargs):
    return {k: v for k, v in kwargs.items() if k in _ACT_KEYS}


class NormAct(nn.Sequential):
    def __init__(self, channels, **kwargs):
        super().__init__()
        self.add_module('norm', norm_layer(channels, **_norm_kwargs(kwargs)))
        self.add_module('act', act_layer(**_act_kwargs(kwargs)))


class NormActConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, **kwargs):
        super().__init__()
        self.add_module('norm_act', NormAct(in_channels, **kwargs))
        self.add_module('conv', conv(in_channels, out_channels,
                                     kernel_size=kernel_size))


class ResBlock(nn.Module):
    def __init__(self, channels, **kwargs):
        super().__init__()
        self.conv1 = NormActConv(channels, channels, **kwargs)
        self.conv2 = NormActConv(channels, channels, **kwargs)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        return x


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.add_module('pre',  NormActConv(in_channels, out_channels, **kwargs))
        self.add_module('res',  ResBlock(out_channels, **kwargs))
        self.add_module('post', NormActConv(out_channels, out_channels, **kwargs))


class TrilinearUp(nn.Module):
    """Trilinear upsampling with unit DC gain on every axis."""
    def __init__(self, factor=(1, 2, 2)):
        super().__init__()
        self.factor = tuple(factor)

    def extra_repr(self):
        return f"factor={self.factor}"

    def forward(self, x):
        if all(f == 1 for f in self.factor):
            return x
        return F.interpolate(x, scale_factor=self.factor, mode='trilinear',
                             align_corners=False)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=(1, 2, 2)):
        super().__init__()
        self.up = nn.Sequential(
            TrilinearUp(factor=up),
            conv(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x, skip):
        return self.up(x) + skip


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=(1, 2, 2), **kwargs):
        super().__init__()
        self.up = UpBlock(in_channels, out_channels, up=up)
        self.conv = ConvBlock(out_channels, out_channels, **kwargs)

    def forward(self, x, skip):
        x = self.up(x, skip)
        return self.conv(x)


class RSUNet(nn.Module):
    """Residual Symmetric U-Net.

    Args:
        width: per-level channel counts; ``len(width) - 1`` down/up steps.
        zfactor: per-level z down/up factor. ``None`` means all 1
            (anisotropic). ``[2] * (len(width) - 1)`` is isotropic.
        norm: ``'gn'`` | ``'in'`` | ``'ln'`` | ``'none'``, or ``'auto'`` to
            pick ``'gn'`` when ``group > 0`` and ``'in'`` otherwise, which is
            what the emvision-based models did.
        group: group size for ``norm='gn'``.
        act: ``'ReLU'`` | ``'LeakyReLU'`` | ``'PReLU'`` | ``'ELU'``.
        act_params: extra keyword arguments for the activation module.
    """
    def __init__(self, width, zfactor=None, norm='auto', group=16, eps=1e-5,
                 affine=True, act='ReLU', act_params=None):
        super().__init__()
        assert len(width) > 1
        depth = len(width) - 1

        if zfactor is None:
            zfactor = [1] * depth
        else:
            assert depth == len(zfactor)

        if norm == 'auto':
            norm = 'gn' if group > 0 else 'in'
        self.norm = norm
        self.act = act

        kwargs = dict(norm=norm, group=group, eps=eps, affine=affine,
                      act=act, act_params=act_params)

        self.iconv = ConvBlock(width[0], width[0], **kwargs)

        self.dconvs = nn.ModuleList()
        for d in range(depth):
            self.dconvs.append(nn.Sequential(
                nn.MaxPool3d((zfactor[d], 2, 2)),
                ConvBlock(width[d], width[d + 1], **kwargs)))

        self.uconvs = nn.ModuleList()
        for d in reversed(range(depth)):
            self.uconvs.append(UpConvBlock(
                width[d + 1], width[d], up=(zfactor[d], 2, 2), **kwargs))

        self.final = NormAct(width[0], **kwargs)

        self.init_weights(act, act_params)

    def forward(self, x):
        x = self.iconv(x)

        skip = list()
        for dconv in self.dconvs:
            skip.append(x)
            x = dconv(x)

        for uconv in self.uconvs:
            x = uconv(x, skip.pop())

        return self.final(x)

    def init_weights(self, act, act_params=None):
        params = dict(act_params or {})
        if act == 'LeakyReLU':
            kwargs = dict(nonlinearity='leaky_relu',
                          a=params.get('negative_slope', 0.01))
        elif act == 'PReLU':
            kwargs = dict(nonlinearity='leaky_relu', a=params.get('init', 0.25))
        else:
            kwargs = dict(nonlinearity='relu')
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, **kwargs)

