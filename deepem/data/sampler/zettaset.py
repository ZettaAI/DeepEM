from __future__ import annotations

import numpy as np

from augmentor import Augment
from dataprovider3 import DataProvider, Dataset, DataSuperset


def get_spec(
    in_spec: dict[str, tuple[int, ...]],
    out_spec: dict[str, tuple[int, ...]],
) -> dict[str, tuple[int, int, int]]:
    spec = {}

    for key, dims in {**in_spec, **out_spec}.items():
        spatial_dims = dims[-3:]
        spec[key] = spatial_dims
        if key in out_spec:
            spec[f"{key}_mask"] = spatial_dims

    return spec


class Sampler:
    def __init__(
        self,
        data: dict[str, dict[str, np.ndarray]],
        spec: dict[str, tuple[int, int, int]],
        is_train: bool,
        aug: Augment | None = None,
        prob: dict[str, float] | None = None,
        zettaset_specs: dict[str, dict] | None = None,
        **kwargs,
    ):
        self.is_train = is_train
        self.dataprovider = self.build_dataprovider(data, spec, aug, prob, zettaset_specs)

    def __call__(self) -> dict[str, np.ndarray]:
        sample = self.dataprovider()
        return self.postprocess(sample)

    def postprocess(
        self, sample: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        sample = Augment.to_tensor(sample)
        return self.convert_to_float32(sample)

    def convert_to_float32(
        self, sample: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        return {k: v.astype(np.float32) for k, v in sample.items()}

    def build_dataprovider(
        self,
        data: dict[str, dict[str, np.ndarray]],
        spec: dict[str, tuple[int, int, int]],
        aug: Augment | None = None,
        prob: dict[str, float] | None = None,
        zettaset_specs: dict[str, dict] | None = None,
    ) -> DataProvider:
        dp = DataProvider(spec)

        for key, dataset_data in data.items():
            if zettaset_specs and key in zettaset_specs:
                dataset = self.build_datasuperset(key, dataset_data, spec)
            else:
                dataset = self.build_dataset(key, dataset_data, spec)

            dp.add_dataset(dataset)

        dp.set_augment(aug)
        dp.set_imgs(["input"])
        dp.set_segs(["affinity", "long_range", "embedding"])

        if prob:
            weights = [prob[k] for k in data]
            dp.set_sampling_weights(p=weights)
        else:
            dp.set_sampling_weights(p=None)

        print(dp)
        return dp

    def build_datasuperset(
        self,
        tag: str,
        data: dict[str, dict[str, np.ndarray]],
        spec: dict[str, tuple[int, int, int]],
    ) -> DataSuperset:
        """Create a DataSuperset from the given data."""
        dset = DataSuperset(tag=tag)

        for key, value in data.items():
            dset.add_dataset(self.build_dataset(key, value, spec))

        return dset

    def build_dataset(
        self,
        tag: str,
        data: dict[str, np.ndarray],
        spec: dict[str, tuple[int, int, int]],
    ) -> Dataset:
        """Create a Dataset from the given data and specification."""
        dset = Dataset(tag=tag)

        for key in spec.keys():
            if key.endswith("_mask"):
                dset.add_mask(key=key, data=data[key], loc=True)
            else:
                dset.add_data(key=key, data=data[key])

        return dset
