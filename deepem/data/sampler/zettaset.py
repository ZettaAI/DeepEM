from __future__ import annotations

import numpy as np

from augmentor import Augment
from dataprovider3 import DataProvider, Dataset, DataSuperset


def get_spec(
    in_spec: dict[str, tuple[int, ...]],
    out_spec: dict[str, tuple[int, ...]],
    extra_spec: dict[str, tuple[int, ...]] | None = None,
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
    # Extra spec
    if extra_spec:
        for k, v in extra_spec.items():
            spec[k] = tuple(v[-3:])
    return spec


class Sampler(object):
    def __init__(
        self,
        data: dict[str, dict[str, np.ndarray]],
        spec: dict[str, tuple[int, int, int]],
        is_train: bool,
        aug: Augment | None = None,
        prob: dict[str, float] | None = None,
        zettaset_specs: dict[str, dict] | None = None,
    ):
        self.is_train = is_train
        self.build(data, spec, aug, prob, zettaset_specs)

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
        zettaset_specs: dict[str, dict] | None = None,
    ) -> None:
        """
        Builds the data provider with datasets, augmentation, and sampling weights.
        """
        self.dataprovider = DataProvider(spec)

         # Add datasets to the data provider
        for key, value in data.items():
            build_method = (
                self.build_datasuperset
                if zettaset_specs and key in zettaset_specs
                else self.build_dataset
            )
            self.dataprovider.add_dataset(build_method(key, value, spec))

        # Set augmentation, image types, and segmentation types
        self.dataprovider.set_augment(aug)
        self.dataprovider.set_imgs(["input"])
        self.dataprovider.set_segs(["affinity", "long_range", "embedding"])

        # Initialize sampling weights (even if prob is None)
        sampling_weights = [prob[k] for k in data.keys()] if prob else None
        self.dataprovider.set_sampling_weights(p=sampling_weights)

        print(self.dataprovider)

    def build_datasuperset(
        self,
        tag: str,
        data: dict[str, dict[str, np.ndarray]],
        spec: dict[str, tuple[int, int, int]],
    ) -> DataSuperset:
        """Create a DataSuperset from the given data."""
        dset = DataSuperset(tag=tag)

        # Add datasets to the DataSuperset
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

        # Iterate over the spec dictionary to add data and masks to the dataset
        for key, _ in spec.items():
            if key.endswith("_mask"):
                dset.add_mask(key=key, data=data[key], loc=True)
            else:
                dset.add_data(key=key, data=data[key])

        return dset
