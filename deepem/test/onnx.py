import numpy as np
import os
import torch

import dataprovider3 as dp3

from deepem.test.option import Options
from deepem.test.utils import *


def batchnorm3d_to_instancenorm3d(model):
    count = 0
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recursion
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


def dummy_input(opt, device='cpu'):
    inputs = {}
    for k in sorted(opt.in_spec):
        size = (1,) + tuple(opt.in_spec[k])
        dummy = torch.randn(*size, device=device)
        inputs[k] = dummy.to(device)
    return inputs


def save_onnx(opt):
    opt.device = 'cpu'
    onnx_model = load_model(opt)
    onnx_model.to('cpu')
    onnx_model, count = batchnorm3d_to_instancenorm3d(onnx_model)
    print(f'Replaced {count} BatchNorm3d layers to InstanceNorm3d layers.')    
    fname = os.path.join(opt.model_dir, "model{}.onnx".format(opt.chkpt_num))
    torch.onnx.export(onnx_model, dummy_input(opt), fname,
                      verbose=False,
                      export_params=True,
                      opset_version=10,
                      input_names=["input"],
                      output_names=["output"])


if __name__ == "__main__":
    # Options
    opt = Options().parse()

    # Run ONNX conversion.
    print(f"Running ONNX conversion: {opt.exp_name} @ {opt.chkpt_num} iters.")
    save_onnx(opt)
