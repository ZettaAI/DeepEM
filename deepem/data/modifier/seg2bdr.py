import numpy as np
import torch

import datatools


class Modifier:
    def __init__(self, key: str = "boundary", create_border: bool = True):
        self.key = key
        self.create_border = create_border

    def __call__(self, sample):
        if self.key in sample:            
            raw = sample[self.key]
            if self.create_border:
                seg = raw[0, 0, :, :, :].cpu().numpy().astype(np.uint32)
                seg = datatools.create_border(seg).astype(np.int32)
                seg = torch.from_numpy(seg).to(raw.device, dtype=raw.dtype)
                raw = seg.unsqueeze(0).unsqueeze(0)
            sample[self.key] = (raw == 0).to(raw.device, dtype=raw.dtype)
        return sample
