#!/usr/bin/env python
"""Regression test for rsunet_decode_iso's decoder input ordering.

The decoder reads the concatenation of every non-decoded output. If that
concatenation followed out_spec iteration order, training and inference would
disagree -- deepem/train/option.py walks the class REGISTRY plainly, while
deepem/test/option.py skips vec/long in that loop and appends them afterwards:

    train out_spec: affinity, long_range, mitochondria, myelin, embedding
    test  out_spec: affinity, mitochondria, myelin, embedding, long_range

Both total 29 channels, so the decoder conv has the same shape either way and a
checkpoint loads with no warning -- the outputs are just silently wrong. This
test builds the model from both orderings and asserts they agree.

Run:  python test_decode_concat_order.py
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deepem.models import rsunet_decode_iso

FOV = (32, 32, 32)
CHANNELS = {
    'affinity': 3,
    'long_range': 3,
    'mitochondria': 1,
    'myelin': 1,
    'embedding': 24,
}
# Orders produced by deepem/train/option.py:405 and deepem/test/option.py:222-231.
TRAIN_ORDER = ['affinity', 'long_range', 'mitochondria', 'myelin', 'embedding']
TEST_ORDER = ['affinity', 'mitochondria', 'myelin', 'embedding', 'long_range']


def make_opt(order):
    """Minimal stand-in for the parsed options, with out_spec in `order`."""
    return argparse.Namespace(
        width=[8, 16], depth=2, group=0, act='ReLU',
        scale_init=0.1, onnx=False, crop=None,
        in_spec=dict(input=(1,) + FOV),
        out_spec={k: (CHANNELS[k],) + FOV for k in order},
        decode_head='affinity', decode_kernel=None,
        decode_stop_grad=False, decode_width=0,
    )


def test_concat_order_is_permutation_invariant():
    m_train = rsunet_decode_iso.create_model(make_opt(TRAIN_ORDER))
    m_test = rsunet_decode_iso.create_model(make_opt(TEST_ORDER))

    # Names are order-independent, so the state dict transfers cleanly. This is
    # exactly why the bug was silent: nothing here complains.
    m_test.load_state_dict(m_train.state_dict())
    m_train.eval()
    m_test.eval()

    torch.manual_seed(0)
    x = torch.randn(1, 1, *FOV)
    with torch.no_grad():
        a = m_train(x)['affinity']
        b = m_test(x)['affinity']

    assert torch.allclose(a, b, atol=1e-6), (
        "decoder output depends on out_spec ordering -- the concat order is not "
        "canonical, so inference would silently permute the decoder's inputs"
    )
    print("[ok] decoder output is identical under train- and test-side out_spec order")


def test_concat_order_is_sorted():
    block = rsunet_decode_iso.create_model(make_opt(TRAIN_ORDER)).out
    expected = ['embedding', 'long_range', 'mitochondria', 'myelin']
    assert block.primary_keys == expected, \
        f"expected {expected}, got {block.primary_keys}"

    decode_in = sum(CHANNELS[k] for k in expected)
    weight = block.affinity.conv.weight
    assert weight.shape[1] == decode_in == 29, \
        f"decoder in_channels {weight.shape[1]} != {decode_in}"
    print(f"[ok] primary_keys sorted {expected}, decoder in_channels = {decode_in}")


def test_stop_grad_isolates_primary_heads():
    for stop_grad in (False, True):
        opt = make_opt(TRAIN_ORDER)
        opt.decode_stop_grad = stop_grad
        model = rsunet_decode_iso.create_model(opt)
        model.zero_grad()
        model(torch.randn(1, 1, *FOV))['affinity'].sum().backward()
        grad = model.out.long_range.conv.weight.grad
        reached = grad is not None and grad.abs().sum().item() > 0
        assert reached == (not stop_grad), (
            f"decode_stop_grad={stop_grad}: affinity gradient "
            f"{'reached' if reached else 'did not reach'} the long_range head"
        )
        print(f"[ok] decode_stop_grad={stop_grad}: primary heads "
              f"{'receive' if reached else 'are isolated from'} the affinity gradient")


def test_onnx_tuple_follows_out_spec_order():
    # OnnxModel zips the tuple against scan_spec.keys(), and export_onnx sets
    # scan_spec = out_spec -- so the tuple must stay in out_spec order, NOT
    # sorted order like the concat.
    for order in (TRAIN_ORDER, TEST_ORDER):
        opt = make_opt(order)
        opt.onnx = True
        model = rsunet_decode_iso.create_model(opt)
        assert model.out.keys == order, f"expected {order}, got {model.out.keys}"
        with torch.no_grad():
            out = model(torch.randn(1, 1, *FOV))
        assert isinstance(out, tuple) and len(out) == len(order)
        widths = [t.shape[1] for t in out]
        assert widths == [CHANNELS[k] for k in order], \
            f"tuple widths {widths} do not match {order}"
    print("[ok] ONNX tuple follows out_spec order on both sides")


if __name__ == "__main__":
    test_concat_order_is_permutation_invariant()
    test_concat_order_is_sorted()
    test_stop_grad_isolates_primary_heads()
    test_onnx_tuple_follows_out_spec_order()
    print("\nAll tests passed.")
