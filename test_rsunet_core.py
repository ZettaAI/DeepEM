#!/usr/bin/env python
"""Tests for deepem/models/core/rsunet.py and the v2 isotropic models.

The emvision comparisons pin down the bug this core was written to fix:
emvision's BilinearUp builds its frozen kernel from the x/y axes only, so with
a z factor of 2 the z taps are all equal -- no interpolation along z, and a
2x gain instead of 1x.
"""

import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deepem.models.core.rsunet import RSUNet, TrilinearUp, check_onnx_opset
from deepem.utils.py_utils import load_module


def test_trilinear_shapes():
    for factor in [(1, 2, 2), (2, 2, 2), (2, 4, 4)]:
        up = TrilinearUp(factor)
        y = up(torch.randn(1, 3, 8, 8, 8))
        assert tuple(y.shape[-3:]) == tuple(8 * f for f in factor), (factor, y.shape)
    print("OK  shapes")


def test_trilinear_z_is_interpolated():
    up = TrilinearUp((2, 2, 2))
    x = torch.zeros(1, 1, 4, 8, 8)
    x[0, 0, :, 4, 4] = torch.tensor([1., 2., 3., 4.])
    z = up(x)[0, 0, :, 8, 8]

    # Adjacent output slices must differ -- the broken kernel emitted pairs.
    dup = [i for i in range(0, 8, 2) if torch.isclose(z[i], z[i + 1])]
    assert not dup, f"duplicated z slice pairs at {dup}: {z.tolist()}"

    # Interior must be a straight line: constant first difference.
    # z[0] and z[-1] are edge-replicated, so the ramp is z[1:7].
    d = z[2:7] - z[1:6]
    assert torch.allclose(d, d[0].expand_as(d), atol=1e-6), z.tolist()
    print("OK  z interpolation")


def test_trilinear_unit_dc_gain():
    for factor in [(1, 2, 2), (2, 2, 2)]:
        y = TrilinearUp(factor)(torch.ones(1, 2, 8, 8, 8))
        assert torch.allclose(y, torch.ones_like(y), atol=1e-6), \
            f"factor {factor}: DC gain {y.mean().item()}"
    print("OK  unit DC gain")


def test_anisotropic_matches_emvision():
    """z factor 1 was never broken; the interior must agree numerically."""
    from emvision.models.layers import BilinearUp

    x = torch.randn(1, 4, 8, 16, 16)
    old = BilinearUp(4, 4, factor=(1, 2, 2))(x)
    new = TrilinearUp((1, 2, 2))(x)
    assert old.shape == new.shape
    # conv_transpose3d truncates its kernel at the border; interpolate
    # replicates the edge. Everything inside must agree to float32 rounding
    # (the two accumulate in a different order).
    interior = (slice(None), slice(None), slice(None), slice(1, -1), slice(1, -1))
    diff = (old[interior] - new[interior]).abs().max().item()
    assert diff < 1e-6, f"interior differs by {diff}"
    print("OK  anisotropic parity with emvision")


def test_state_dict_keys_match_emvision():
    """New keys must be the old ones minus the dropped BilinearUp buffers."""
    from emvision.models import rsunet_act_gn, rsunet_act_in

    cases = [
        (lambda: rsunet_act_gn(width=[16, 32, 64, 128], zfactor=[2, 2, 2],
                               group=16, act='ELU'),
         lambda: RSUNet(width=[16, 32, 64, 128], zfactor=[2, 2, 2],
                        norm='gn', group=16, act='ELU')),
        (lambda: rsunet_act_in(width=[16, 32, 64, 128], zfactor=[2, 2, 2],
                               act='ELU'),
         lambda: RSUNet(width=[16, 32, 64, 128], zfactor=[2, 2, 2],
                        norm='in', act='ELU')),
    ]
    for build_old, build_new in cases:
        old, new = build_old().state_dict(), build_new().state_dict()
        dropped = set(old) - set(new)
        assert dropped == {f'uconvs.{i}.up.up.0.weight' for i in range(3)}, dropped
        assert not set(new) - set(old), set(new) - set(old)
        bad = [k for k in new if old[k].shape != new[k].shape]
        assert not bad, bad
    print("OK  state_dict key parity")


def test_norm_variants():
    x = torch.randn(1, 16, 8, 16, 16)
    expected = {
        'gn': torch.nn.GroupNorm,
        'in': torch.nn.InstanceNorm3d,
        'ln': torch.nn.GroupNorm,
        'none': torch.nn.Identity,
    }
    for norm, cls in expected.items():
        net = RSUNet(width=[16, 32], zfactor=[2], norm=norm, group=8)
        assert isinstance(net.final.norm, cls), (norm, type(net.final.norm))
        if norm == 'ln':
            assert net.final.norm.num_groups == 1
        elif norm == 'gn':
            assert net.final.norm.num_groups == 16 // 8
        assert net(x).shape == x.shape
    # 'auto' reproduces what the emvision-based models did.
    assert RSUNet(width=[16, 32], group=16).norm == 'gn'
    assert RSUNet(width=[16, 32], group=0).norm == 'in'
    print("OK  norm variants")


def _opt(**kw):
    base = dict(
        width=[16, 32, 64], depth=3, group=16, group_eps=1e-5, act='ELU',
        norm='auto', crop=None, onnx=False, scale_init=1.0,
        in_spec={'input': (1, 32, 64, 64)},
        out_spec={'affinity': (3, 32, 64, 64)},
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_v2_models_forward():
    x = torch.randn(1, 1, 32, 64, 64)
    for name in ['rsunet_act_iso', 'rsunet_multi_io_iso']:
        mod = load_module('model', f'deepem/models/v2/{name}.py')
        model = mod.create_model(_opt()).eval()
        with torch.no_grad():
            out = model(x)
        assert set(out) == {'affinity'}, out.keys()
        assert tuple(out['affinity'].shape) == (1, 3, 32, 64, 64), out['affinity'].shape
        assert torch.isfinite(out['affinity']).all()
        print(f"OK  {name} forward")


def test_onnx_opset_guard():
    check_onnx_opset(_opt(onnx=False, opset_version=10))  # not exporting: fine
    check_onnx_opset(_opt(onnx=True, opset_version=11))
    for name in ['rsunet_act_iso', 'rsunet_multi_io_iso']:
        mod = load_module('model', f'deepem/models/v2/{name}.py')
        try:
            mod.create_model(_opt(onnx=True, opset_version=10))
        except ValueError as e:
            assert 'opset' in str(e)
        else:
            raise AssertionError(f"{name}: opset 10 export was not rejected")
    print("OK  ONNX opset guard")


if __name__ == '__main__':
    test_trilinear_shapes()
    test_trilinear_z_is_interpolated()
    test_trilinear_unit_dc_gain()
    test_anisotropic_matches_emvision()
    test_state_dict_keys_match_emvision()
    test_norm_variants()
    test_v2_models_forward()
    test_onnx_opset_guard()
    print("\nall passed")
