#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch

from deepem.test.option import Options
from deepem.test.utils import load_model


def batchnorm3d_to_instancenorm3d(model):
    count = 0
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name], cnt = batchnorm3d_to_instancenorm3d(module)
            count += cnt

        if isinstance(module, torch.nn.BatchNorm3d):
            layer_old = module
            layer_new = torch.nn.InstanceNorm3d(module.num_features, 
                                                affine=module.affine, 
                                                track_running_stats=False)
            layer_new.weight = module.weight
            layer_new.bias = module.bias
            model._modules[name] = layer_new
            count += 1    
    return model, count


def dummy_input(opt, device='cpu'):
    inputs = {}
    for k in sorted(opt.in_spec):
        size = (1,) + tuple(opt.in_spec[k])
        inputs[k] = torch.randn(*size, device=device)
    return inputs


if __name__ == "__main__":
    # Options
    opt = Options().parse()
    opt.onnx = True

    # CPU or GPU
    opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_model = load_model(opt)
    torch_model, count = batchnorm3d_to_instancenorm3d(torch_model)
    print(f"replaced {count} BatchNorm3d layer to InstanceNorm3d layer.")
    fname = os.path.join(opt.model_dir, "model{}.onnx".format(opt.chkpt_num))
    
    # Run ONNX conversion.
    torch.onnx.export(
        torch_model,
        (dummy_input(opt, device=opt.device), {}),
        fname,
        verbose=False,
        export_params=True,
        opset_version=16,
        input_names=["input"],
        output_names=["output"]
    )
    print(f"Relative ONNX filepath: {fname}")
