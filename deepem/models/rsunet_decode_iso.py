"""
Isotropic RSUNet where one output is decoded from the *other* outputs.

Topology (vs. the parallel-heads OutputBlock of rsunet_act_iso):

    core ──┬─[conv]──────► long_range   (3)
           ├─[conv]──────► mitochondria (1)
           ├─[conv]──────► myelin       (1)
           └─[conv+Scale]► embedding    (24)
                               │
                     concat along channels
                               │
                     [.detach() if stop_grad]
                               │
                          [decoder]
                               │
                               ▼
                          affinity (3)

Every output except `--decode_head` reads the U-Net features exactly as in
rsunet_act_iso; the decoded head reads the concatenation of all of them. With
--decode_stop_grad the concatenation is detached, so the other outputs are
shaped only by their own losses and the decoder becomes a pure read-out.

Raw logits are concatenated -- no sigmoid. The BCE margin (loss.py, with
inverse=False) masks out voxels once sigmoid(logit) >= 1 - margin1, so a large
fraction sit at saturation by construction; squashing would collapse "barely
confident" and "very confident" onto the same value, discarding resolution
precisely near boundaries. No normalization either: the embedding's absolute
scale is calibrated against delta_d by the metric loss, and a per-channel norm
would rescale embedding space anisotropically.

Submodule names match rsunet_act_iso exactly, so an rsunet_act_iso checkpoint
loads key-for-key via --pretrain, leaving only the decoder freshly initialized.

INVARIANT -- the concat order is part of what a checkpoint means. The decoder's
input channels are the primary outputs in **sorted key order**, deliberately not
out_spec iteration order: deepem/train/option.py walks the class REGISTRY
plainly, while deepem/test/option.py skips vec/long in that loop and appends
them afterwards, so the two sides disagree:

    train out_spec: affinity, long_range, mitochondria, myelin, embedding
    test  out_spec: affinity, mitochondria, myelin, embedding, long_range

Both total the same width, so a checkpoint would load without any warning and
inference would silently return garbage with every decoder weight on the wrong
channel. Sorting makes the layout identical on both sides. Changing the set of
heads changes the decoder's in_channels, so load_state_dict fails loudly rather
than misaligning -- but do not "tidy" this back into out_spec order.
"""
import torch
import torch.nn as nn

import emvision

from deepem.models.layers import Conv, Crop, Scale


def create_model(opt):
    if opt.width:
        width = opt.width
        depth = len(width)
    else:
        width = [16,32,64,128,256,512]
        depth = opt.depth

    # Isotropic up & down sampling
    zfactor = [2] * (depth - 1)

    if opt.group > 0:
        # Group normalization
        core = emvision.models.rsunet_act_gn(
            width=width[:depth],
            zfactor=zfactor,
            group=opt.group,
            act=opt.act,
        )
    else:
        # Instance normalization
        core = emvision.models.rsunet_act_in(
            width=width[:depth],
            zfactor=zfactor,
            act=opt.act,
        )
    return Model(core, opt.in_spec, opt.out_spec, width[0], opt, crop=opt.crop,
                 onnx=opt.onnx, scale_init=opt.scale_init)


def build_decoder(in_channels, out_channels, opt):
    """Build the module that decodes one output from the others.

    This is the only thing to edit when the decoder generalizes. Contract:
    consumes the concatenated primary outputs (raw logits, `in_channels`),
    emits `out_channels` logits at the same spatial size.
    """
    kernel = tuple(getattr(opt, 'decode_kernel', None) or (5, 5, 5))
    width = getattr(opt, 'decode_width', 0)
    if width > 0:
        return nn.Sequential(
            Conv(in_channels, width, kernel, bias=True),
            nn.ReLU(inplace=True),
            Conv(width, out_channels, (1, 1, 1), bias=True),
        )
    return Conv(in_channels, out_channels, kernel, bias=True)


class InputBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(InputBlock, self).__init__()
        self.add_module('conv', Conv(in_channels, out_channels, kernel_size))


class OutputBlock(nn.Module):
    """Parallel heads off the trunk, plus one head decoded from their concat."""
    def __init__(self, in_channels, out_spec, kernel_size, opt, onnx=False,
                 scale_init=1.0):
        super(OutputBlock, self).__init__()
        decode_key = getattr(opt, 'decode_head', None) or 'affinity'
        assert decode_key in out_spec, \
            f"rsunet_decode_iso requires a '{decode_key}' output in out_spec"
        self.onnx = onnx
        self.stop_grad = getattr(opt, 'decode_stop_grad', False)
        self.decode_key = decode_key
        # Preserve out_spec ordering for the ONNX tuple output
        self.keys = list(out_spec.keys())
        # Sorted, NOT out_spec order: train and test build out_spec differently
        # (deepem/test/option.py appends vec/long last), so iteration order would
        # permute the decoder's input channels between training and inference.
        self.primary_keys = sorted(k for k in out_spec if k != decode_key)
        assert self.primary_keys, "nothing to decode from"

        # Primary heads read the U-Net features (names match rsunet_act_iso).
        for k in self.primary_keys:
            out_channels = out_spec[k][-4]
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

        # Decoded head reads the concatenated primary outputs.
        decode_in = sum(out_spec[k][-4] for k in self.primary_keys)
        self.add_module(decode_key,
            build_decoder(decode_in, out_spec[decode_key][-4], opt))

    def forward(self, x):
        outputs = {k: getattr(self, k)(x) for k in self.primary_keys}
        feat = torch.cat([outputs[k] for k in self.primary_keys], dim=1)
        if self.stop_grad:
            feat = feat.detach()
        outputs[self.decode_key] = getattr(self, self.decode_key)(feat)
        if self.onnx:
            return tuple(outputs[k] for k in self.keys)
        return outputs


class Model(nn.Sequential):
    """
    Residual Symmetric U-Net with an output decoded from the other outputs.
    """
    def __init__(self, core, in_spec, out_spec, out_channels, opt,
                 io_kernel=(5,5,5), crop=None, onnx=False, scale_init=1.0):
        super(Model, self).__init__()

        assert len(in_spec)==1, "model takes a single input"
        in_channels = 1

        self.add_module('in', InputBlock(in_channels, out_channels, io_kernel))
        self.add_module('core', core)
        self.add_module('out', OutputBlock(out_channels, out_spec, io_kernel, opt,
                                           onnx=onnx, scale_init=scale_init))
        if crop is not None:
            self.add_module('crop', Crop(crop))
