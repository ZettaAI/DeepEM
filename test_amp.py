#!/usr/bin/env python
"""Tests for deepem/train/amp.py and the mixed-precision path of the training
model wrapper.

These run on CPU, where autocast supports bf16 (and fp16 for the scaler
logic), so they pin the plumbing: which dtype the network actually runs in,
that losses are computed in fp32, that fp32 training is untouched, and that the
fp16 loss scale survives a checkpoint. Numerical parity and speed on a real GPU
are checked by scripts/amp_bench.py.

The regression this guards against: AmpModel opened a dtype-less
``torch.cuda.amp.autocast()`` (fp16) inside run.py's bf16 context, so
``--mixed_precision bf16`` trained in fp16 without loss scaling.
"""

import os
import sys
import tempfile
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import deepem.loss as loss
from deepem.train import utils as train_utils
from deepem.train.amp import Precision
from deepem.train.model import AmpModel, Model
from deepem.utils.py_utils import load_module

SHAPE = (16, 32, 32)


def _opt(**kw):
    base = dict(
        width=[8, 16, 32], depth=3, group=0, group_eps=1e-5, act='ReLU',
        norm='auto', crop=None, onnx=False, scale_init=1.0,
        in_spec={'input': (1,) + SHAPE},
        out_spec={'affinity': (3,) + SHAPE, 'myelin': (1,) + SHAPE},
        pretrain=None, mixed_precision=None, channels_last=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _model(seed=0, **kw):
    opt = _opt(**kw)
    torch.manual_seed(seed)
    net = load_module('model', 'deepem/models/v2/rsunet_multi_io_iso.py').create_model(opt)
    criteria = {k: loss.BCELoss(size_average=True) for k in opt.out_spec}
    return Model(net, criteria, opt)


def _sample(seed=1):
    g = torch.Generator().manual_seed(seed)
    s = {'input': torch.rand((1, 1) + SHAPE, generator=g)}
    for k, c in (('affinity', 3), ('myelin', 1)):
        s[k] = (torch.rand((1, c) + SHAPE, generator=g) > 0.5).float()
        s[k + '_mask'] = torch.ones((1, c) + SHAPE)
    return s


def _conv_dtypes(model):
    """Output dtype of every Conv3d during the next forward."""
    seen = []
    hooks = [m.register_forward_hook(lambda m, i, o: seen.append(o.dtype))
             for m in model.modules() if isinstance(m, torch.nn.Conv3d)]
    return seen, hooks


def test_modes():
    assert Precision(None).mode == 'fp32' and not Precision(None).enabled
    assert Precision('bf16').dtype is torch.bfloat16 and Precision('bf16').scaler is None
    assert Precision('fp16', device_type='cpu').scaler is not None
    with pytest.raises(ValueError):
        Precision('fp8')


def test_ampmodel_no_longer_forces_fp16():
    assert AmpModel is Model


def test_bf16_runs_network_in_bf16_and_loss_in_fp32():
    model = _model(mixed_precision='bf16')
    seen, hooks = _conv_dtypes(model)
    losses, nmasks, preds = model(_sample())
    for h in hooks:
        h.remove()
    assert seen and set(seen) == {torch.bfloat16}, set(seen)
    assert all(p.dtype == torch.float32 for p in preds.values())
    assert all(l.dtype == torch.float32 for l in losses.values())


def test_outer_fp16_context_cannot_leak_into_bf16():
    """Autocast is set by the wrapper, whatever context the caller is in."""
    model = _model(mixed_precision='bf16')
    seen, hooks = _conv_dtypes(model)
    with torch.autocast('cpu', dtype=torch.float16):
        model(_sample())
    for h in hooks:
        h.remove()
    assert set(seen) == {torch.bfloat16}, set(seen)


def test_fp32_path_is_unchanged():
    """No autocast, and exactly what calling the net and the losses directly gives."""
    model = _model()
    sample = _sample()
    seen, hooks = _conv_dtypes(model)
    losses, _, preds = model(sample)
    for h in hooks:
        h.remove()
    assert set(seen) == {torch.float32}
    raw = model.model(sample['input'])
    for k in model.out_spec:
        assert torch.equal(preds[k], raw[k])
        ref, _ = model.criteria[k](raw[k], sample[k], sample[k + '_mask'])
        assert torch.equal(losses[k].squeeze(0), ref)


def test_bf16_loss_close_to_fp32():
    sample = _sample()
    ref, _, _ = _model()(sample)
    low, _, _ = _model(mixed_precision='bf16')(sample)
    for k in ref:
        rel = ((low[k] - ref[k]).abs() / ref[k].abs()).item()
        assert rel < 2e-2, (k, rel)


def _grads(sample, mode=None):
    model = _model(mixed_precision=mode)
    losses, _, _ = model(sample)
    sum(losses.values()).sum().backward()
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])


def test_bf16_gradients_within_bf16_noise_floor():
    """At random init the gradient direction is very sensitive: in pure fp32,
    perturbing the input by bf16's relative precision (2^-8) already moves it
    to cosine ~0.92. bf16 training must not do worse than that."""
    sample = _sample()
    ref = _grads(sample)
    cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    torch.manual_seed(5)
    noisy = dict(sample, input=sample['input'] * (1 + 2 ** -8 * torch.randn_like(sample['input'])))
    floor = cos(ref, _grads(noisy))
    got = cos(ref, _grads(sample, 'bf16'))
    assert got > floor - 0.03, (got, floor)


def test_channels_last_matches_default_layout():
    sample = _sample()
    ref, _, ref_preds = _model()(sample)
    model = _model(channels_last=True).to(memory_format=torch.channels_last_3d)
    got, _, preds = model(sample)
    for k in ref:
        assert torch.allclose(preds[k], ref_preds[k], atol=1e-5), k
        assert torch.allclose(got[k], ref[k], rtol=1e-5), k


def test_backward_step_plain_and_scaled():
    for mode in (None, 'bf16', 'fp16'):
        p = Precision(mode, device_type='cpu')
        w = torch.nn.Parameter(torch.ones(3))
        opt = torch.optim.SGD([w], lr=0.1)
        assert p.backward_step((w * 2).sum(), opt) is True
        assert torch.allclose(w.detach(), torch.full((3,), 0.8)), (mode, w)


def test_fp16_step_skipped_on_inf():
    p = Precision('fp16', device_type='cpu')
    w = torch.nn.Parameter(torch.ones(3))
    opt = torch.optim.SGD([w], lr=0.1)
    scale = p.loss_scale
    assert p.backward_step((w * float('inf')).sum(), opt) is False
    assert torch.equal(w.detach(), torch.ones(3))
    assert p.loss_scale < scale


def test_loss_scale_survives_checkpoint():
    model = _model(mixed_precision='fp16')
    p = Precision('fp16', device_type='cpu')
    p.scaler.scale(torch.ones(1))    # the scale tensor is created lazily
    p.scaler.update(1024.0)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with tempfile.TemporaryDirectory() as d:
        train_utils.save_chkpt(model, d, 7, opt, p)
        chkpt = train_utils.load_optimizer_state(opt, d, 7)
        assert chkpt['amp']['mode'] == 'fp16'
        q = Precision('fp16', device_type='cpu')
        q.load_state_dict(chkpt.get('amp'))
        assert q.loss_scale == 1024.0


def test_checkpoints_without_amp_state_still_load():
    model = _model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with tempfile.TemporaryDirectory() as d:
        train_utils.save_chkpt(model, d, 3, opt)            # fp32: no 'amp' key
        chkpt = train_utils.load_optimizer_state(opt, d, 3)
        assert 'amp' not in chkpt
        q = Precision('fp16', device_type='cpu')
        q.load_state_dict(chkpt.get('amp'))                 # old checkpoint -> default scale
        assert q.loss_scale == 2.0 ** 16
        Precision('bf16').load_state_dict({'mode': 'fp16', 'scaler': {'scale': 8.0}})  # ignored
