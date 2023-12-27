from __future__ import annotations

import numpy as np

from augmentor import Augment
from dataprovider3 import DataProvider, Dataset


def get_spec(
    in_spec: dict[str, tuple[int, ...]],
    out_spec: dict[str, tuple[int, ...]],
) -> dict[str, tuple[int, int, int]]:
    spec = dict()
    # Input spec
    for k, v in in_spec.items():
        spec[k] = tuple(v[-3:])
    # Output spec
    for k, v in out_spec.items():
        dim = tuple(v[-3:])
        spec[k] = dim
        spec[k+'_mask'] = dim
    return spec


class Sampler(object):
    def __init__(
        self,
        data: dict[str, dict[str, np.ndarray]],
        spec: dict[str, tuple[int, int, int]],
        is_train: bool,
        aug: Augment | None = None,
        prob: dict[str, float] | None = None,
    ):
        self.is_train = is_train
        self.build(data, spec, aug, prob)

    def __call__(self) -> dict[str, np.ndarray]:
        sample = self.dataprovider()
        return self.postprocess(sample)

    def postprocess(
        self, sample: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        sample = Augment.to_tensor(sample)
        return self.to_float32(sample)

    def to_float32(
        self, sample: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        for k, v in sample.items():
            sample[k] = v.astype('float32')
        return sample

    def build(
        self,
        data: dict[str, dict[str, np.ndarray]],
        spec: dict[str, tuple[int, int, int]],
        aug: Augment | None = None,
        prob: dict[str, float] | None = None,
    ) -> None:
        dp = DataProvider(spec)
        keys = data.keys()
        for key in keys:
            dp.add_dataset(self.build_dataset(key, data[key], spec))
        dp.set_augment(aug)
        dp.set_imgs(["input"])
        dp.set_segs(["affinity", "long_range", "embedding"])
        prob = [prob[k] for k in keys] if prob is not None else prob
        dp.set_sampling_weights(p=prob)
        self.dataprovider = dp
        print(dp)

    def build_dataset(
        self,
        tag: str,
        data: dict[str, np.ndarray],
        spec: dict[str, tuple[int, int, int]],
    ) -> Dataset:
        """Create a Dataset."""
        dset = Dataset(tag=tag)

        for key in spec.keys():
            if key.endswith("_mask"):
                dset.add_mask(key=key, data=data[key], loc=True)
            else:
                dset.add_data(key=key, data=data[key])

        return dset
