from __future__ import annotations

import argparse
import copy
import os
import torch

from deepem.test.utils import load_model


def batchnorm3d_to_instancenorm3d(
    model: torch.nn.Module
) -> tuple[torch.nn.Module, int]:
    count = 0
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name], cnt = batchnorm3d_to_instancenorm3d(module)
            count += cnt

        if isinstance(module, torch.nn.BatchNorm3d):
            layer_new = torch.nn.InstanceNorm3d(module.num_features,
                                                affine=module.affine,
                                                track_running_stats=False)
            layer_new.weight = module.weight
            layer_new.bias = module.bias
            model._modules[name] = layer_new
            count += 1
    return model, count


def dummy_input(
    spec: dict[str, tuple[int, ...]],
    device: str = 'cpu'
) -> dict[str, torch.Tensor]:
    """Generate a random input."""
    inputs = {}
    for k in sorted(spec):
        size = (1,) + tuple(spec[k])
        inputs[k] = torch.randn(*size, device=device)
    return inputs


def export_onnx(
    opt: argparse.Namespace,
    chkpt_num: int
) -> None:
    """Export ONNX model."""
    # Prepare options
    onnx_opt = copy.deepcopy(opt)
    onnx_opt.chkpt_num = chkpt_num
    onnx_opt.no_eval = False
    onnx_opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    onnx_opt.onnx = True

    # Supplement missing options, if any
    onnx_opt.scan_spec = onnx_opt.out_spec
    onnx_opt.force_crop = None
    onnx_opt.temperature = None
    onnx_opt.crop = None
    onnx_opt.blend = "bump"

    # Prepare model
    onnx_model = load_model(onnx_opt)
    onnx_model, count = batchnorm3d_to_instancenorm3d(onnx_model)
    print(f"Replaced {count} BatchNorm3d layer to InstanceNorm3d layer.")
    fname = os.path.join(onnx_opt.model_dir, f"model{chkpt_num}.onnx")

    # Arugment passing differs according to the PyTorch version
    args = dummy_input(onnx_opt.in_spec, device=onnx_opt.device)
    if torch.__version__ >= '1.10':
        args = (args, {})

    # Run ONNX conversion
    torch.onnx.export(
        onnx_model,
        args,
        fname,
        verbose=False,
        export_params=True,
        opset_version=onnx_opt.opset_version,
        input_names=["input"],
        output_names=["output"]
    )
    print(f"Relative ONNX filepath: {fname}")
