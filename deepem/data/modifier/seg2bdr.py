import numpy as np
import torch

import datatools


class Modifier:
    def __init__(self, key: str = "boundary", create_border: bool = False, **kwargs):
        self.key = key
        self.create_border = create_border

    def __call__(self, sample, **kwargs):
        if self.key in sample:
            raw = sample[self.key]

            # Iterate over the batch dimension
            for b in range(raw.shape[0]):
                if self.create_border:
                    seg = raw[b, 0, :, :, :].cpu().numpy().astype(np.uint32)
                    seg = datatools.create_border(seg).astype(np.int32)
                    seg = torch.from_numpy(seg).to(raw.device, dtype=raw.dtype)
                    raw[b, 0, :, :, :] = seg

            sample[self.key] = (raw == 0).to(raw.device, dtype=raw.dtype)

        return sample
