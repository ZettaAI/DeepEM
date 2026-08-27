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
        # Any affinity* head ('affinity', 'affinity_overseg', ...) is a
        # segmentation target; exclude the companion '*_mask' keys.
        seg_keys = [
            k for k in spec
            if k.startswith("affinity") and not k.endswith("_mask")
        ]
        seg_keys += ["long_range", "embedding"]
        dp.set_segs(seg_keys)

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

        # add_mask(loc=True) runs np.flatnonzero over the whole volume and
        # unions the result into Dataset.locs -- an int64 index array with one
        # entry per nonzero mask voxel (1.05 GiB for a hemibrain sample at 8nm,
        # whose masks are all-ones over the unpadded label volume). Every mask
        # key paid that, and each additional union1d allocates ~4x the index
        # array on top of it (concatenate, then sort).
        #
        # The masks are overwhelmingly redundant in practice: load_sample shares
        # one array across lookup names resolving to the same source, and the
        # remaining distinct arrays are usually still identical in content
        # (all-ones over the same label bbox). Masks with equal content
        # contribute an identical index set, so the union is provably a no-op.
        # Compute locs once per distinct mask *content*. Measured on a hemibrain
        # sample at 8nm: peak +5.98 GiB -> +1.05 GiB.
        loc_masks: list[tuple[np.ndarray, int]] = []

        def locs_nnz(mask: np.ndarray) -> int | None:
            """Nonzero count if this mask adds locations, else None."""
            nnz = None
            for seen, seen_nnz in loc_masks:
                if mask is seen:
                    return None
                if mask.shape != seen.shape or mask.dtype != seen.dtype:
                    continue
                # count_nonzero allocates nothing; only fall through to
                # array_equal (which does) when it cannot rule equality out.
                if nnz is None:
                    nnz = int(np.count_nonzero(mask))
                if nnz == seen_nnz and np.array_equal(mask, seen):
                    return None
            return nnz if nnz is not None else int(np.count_nonzero(mask))

        for key in spec.keys():
            if key.endswith("_mask"):
                mask = data[key]
                nnz = locs_nnz(mask)
                if nnz is not None:
                    loc_masks.append((mask, nnz))
                # An all-zero mask contributes no locations. Registering it as
                # one makes the dataset look sampleable and defers the failure
                # to a confusing out-of-range error at sample time -- this is
                # the zero-fill path in multi_zettaset, so it is reachable.
                dset.add_mask(key=key, data=mask,
                              loc=nnz is not None and nnz > 0)
            else:
                dset.add_data(key=key, data=data[key])

        return dset
