"""
Isotropic RSUNet that predicts an embedding, then decodes every other output
(affinity, mitochondria, myelin, ...) from that embedding via a single conv.

Topology (vs. the standard parallel-heads OutputBlock):

    core ── [embedding head] ── embedding ──► metric loss
                                   │
                                   ├─ [conv] ─► affinity
                                   ├─ [conv] ─► mitochondria
                                   └─ [conv] ─► ...

The embedding head reads the U-Net features; every task head reads the
embedding (not the features). With --embed_stop_grad the embedding is detached
before the task heads, so it is shaped only by the metric-learning loss and the
heads act as pure decoders. Default: gradients flow (joint multi-task).

Requires an 'embedding' output in out_spec.
"""
import torch
import torch.nn as nn

from emvision.models import rsunet_act_in, rsunet_act_gn

from deepem.models.layers import Conv, Crop, Scale


def create_model(opt):
    width = opt.width if opt.width else [16, 32, 64, 128, 256, 512]
    depth = len(width) if opt.width else opt.depth
    zfactor = [2] * (depth - 1)
    if opt.group > 0:
        # Group normalization
        core = rsunet_act_gn(width=width[:depth], zfactor=zfactor,
                             group=opt.group, eps=opt.group_eps, act=opt.act)
    else:
        # Instance normalization
        core = rsunet_act_in(width=width[:depth], zfactor=zfactor, act=opt.act)
    decode_kernel = tuple(getattr(opt, 'embed_decode_kernel', None) or (5, 5, 5))
    return Model(core, opt.in_spec, opt.out_spec, width[0], crop=opt.crop,
                 onnx=opt.onnx, scale_init=opt.scale_init,
                 decode_kernel=decode_kernel,
                 stop_grad=getattr(opt, 'embed_stop_grad', False))


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
    """Embedding head + per-task decoding heads that read the embedding."""
    def __init__(self, in_channels, out_spec, kernel_size, decode_kernel,
                 onnx=False, scale_init=1.0, stop_grad=False):
        super().__init__()
        assert 'embedding' in out_spec, \
            "rsunet_embed_iso requires an 'embedding' output in out_spec"
        self.onnx = onnx
        self.stop_grad = stop_grad
        # Preserve out_spec ordering for the ONNX tuple output
        self.keys = list(out_spec.keys())

        embed_channels = out_spec['embedding'][-4]
        # Embedding head reads the U-Net features.
        self.embedding = nn.Sequential(
            Conv(in_channels, embed_channels, kernel_size, bias=True),
            Scale(init_value=scale_init),
        )
        # Task heads decode from the embedding (single conv each).
        self.heads = nn.ModuleDict({
            k: Conv(embed_channels, v[-4], decode_kernel, bias=True)
            for k, v in out_spec.items() if k != 'embedding'
        })

    def forward(self, x):
        emb = self.embedding(x)
        feat = emb.detach() if self.stop_grad else emb
        outputs = {'embedding': emb}
        for k, m in self.heads.items():
            outputs[k] = m(feat)
        if self.onnx:
            return tuple(outputs[k] for k in self.keys)
        return outputs


class Model(nn.Sequential):
    """
    Isotropic Residual Symmetric U-Net with embedding-decoding heads.
    """
    def __init__(self, core, in_spec, out_spec, out_channels, io_kernel=(5, 5, 5),
                 decode_kernel=(5, 5, 5), crop=None, onnx=False, scale_init=1.0,
                 stop_grad=False):
        super().__init__()
        self.add_module('in', InputBlock(in_spec, out_channels, io_kernel))
        self.add_module('core', core)
        self.add_module('out',
            OutputBlock(out_channels, out_spec, io_kernel, decode_kernel,
                        onnx=onnx, scale_init=scale_init, stop_grad=stop_grad))
        if crop is not None:
            self.add_module('crop', Crop(crop))
