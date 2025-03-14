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
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Generate random input tensor(s).

    Args:
        spec: Dictionary mapping input names to their shapes
        device: Device to create tensors on

    Returns:
        Single tensor for single input, tuple of tensors for multiple inputs
    """
    tensors = []
    for k in sorted(spec):
        size = (1,) + tuple(spec[k])
        tensors.append(torch.randn(*size, device=device))

    # Return single tensor for backwards compatibility if only one input
    return tensors[0] if len(tensors) == 1 else tuple(tensors)


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

    # Dummy options for inference
    onnx_opt.vec_to = "aff"
    onnx_opt.edges = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]

    # Prepare model
    onnx_model = load_model(onnx_opt)
    onnx_model, count = batchnorm3d_to_instancenorm3d(onnx_model)
    print(f"Replaced {count} BatchNorm3d layer to InstanceNorm3d layer.")
    fname = os.path.join(onnx_opt.model_dir, f"model{chkpt_num}.onnx")

    args = dummy_input(onnx_opt.in_spec, device=onnx_opt.device)

    # For PyTorch >= 1.10, export() expects model_args and model_kwargs separately
    export_args = (args, {}) if torch.__version__ >= '1.10' else args

    # Generate input names based on spec keys
    input_names = sorted(onnx_opt.in_spec.keys())

    torch.onnx.export(
        onnx_model,
        export_args,
        fname,
        verbose=False,
        export_params=True,
        opset_version=onnx_opt.opset_version,
        input_names=input_names,  # dynamic input names based on spec
        output_names=["output"]
    )
    print(f"Relative ONNX filepath: {fname}")
