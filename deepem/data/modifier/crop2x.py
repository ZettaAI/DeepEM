import numpy as np

from deepem.utils.torch_utils import crop_center


class Modifier(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, sample, is_train=False, **kwargs):
        if is_train and (np.random.rand() < 0.5):
            for k, v in sample.items():
                cropsz = (v.shape[-3], v.shape[-2]//2, v.shape[-1]//2)
                sample[k] = crop_center(v, cropsz).contiguous()
        return sample
