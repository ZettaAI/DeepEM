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

    # Additional inputs for assignment
    if "assignment" in out_spec:
        spec[f"assignment_synapse"] = tuple(v[-3:])

    # Output spec
    for k, v in out_spec.items():
        dim = tuple(v[-3:])
        if "assignment" not in k:
            spec[k] = dim
            spec[k+'_mask'] = dim
        else:  # assignment task
            spec[f"assignment_segmentation"] = dim
            spec[f"assignment_synapse_mask"] = dim
    return spec


class Sampler(object):
    def __init__(
        self,
        data: dict[str, dict[str, np.ndarray]],
        spec: dict[str, tuple[int, int, int]],
        is_train: bool,
        aug: Augment | None = None,
        prob: dict[str, float] | None = None
    ):
        self.is_train = is_train
        self.build(data, spec, aug, prob)

    def __call__(self) -> dict[str, np.ndarray]:
        idx = self.dataprovider.random_dataset_idx()
        sample = self.dataprovider.random_sample(idx=idx)
        return self.postprocess(sample, idx)

    def postprocess(
        self, sample: dict[str, np.ndarray], idx: int
    ) -> dict[str, np.ndarray]:
        if "assignment_synapse" in sample:
            img = sample.pop("input")
            syn = sample.pop("assignment_synapse")
            seg = sample.pop("assignment_segmentation")
            msk = sample.pop("assignment_synapse_mask")

            i = self.pick_synapse(syn)
            sample["input"] = self.assignment_input(img, syn, i)
            sample["assignment"] = self.assignment_output(seg, idx, i)
            sample["assignment_mask"] = msk

        sample = Augment.to_tensor(sample)
        return self.to_float32(sample)

    def to_float32(
        self, sample: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        for k, v in sample.items():
            sample[k] = v.astype('float32')
        return sample

    def pick_synapse(self, syn: np.ndarray):
        ids = np.unique(syn)
        ids = ids[ids != 0]

        assert len(ids) > 0, "no ids found in sample"
        return np.random.choice(ids)

    def assignment_input(self, img: np.ndarray, syn: np.ndarray, i: int) -> np.ndarray:
        synapse_mask = (syn == i)

        return np.concatenate((img, synapse_mask), axis=0)

    def assignment_output(self, seg: np.ndarray, idx: int, i: int) -> np.ndarray:
        assert hasattr(self, "extras"), "no extra data found"
        presyn, postsyn = self.extras[idx][str(i)]
        n = len(presyn)
        presyn_mask = seg == presyn[0]
        postsyn_mask = seg == postsyn[0]
        for i in range(1, n):
            presyn_mask = presyn_mask+(seg==presyn[i])
            postsyn_mask = postsyn_mask+(seg==postsyn[i])
        
        return np.concatenate((presyn_mask, postsyn_mask), axis=0)

    def build(
        self,
        data: dict[str, dict[str, np.ndarray]],
        spec: dict[str, tuple[int, int, int]],
        aug: Augment | None,
        prob: dict[str, float] | None
    ) -> None:
        dp = DataProvider(spec)
        keys = data.keys()
        extras = list()
        for key in keys:
            dset, extra = self.build_dataset(key, data[key], spec)
            dp.add_dataset(dset)
            extras.append(extra)
        dp.set_augment(aug)
        dp.set_imgs(["input"])
        dp.set_segs([
            "affinity",
            "long_range",
            "assignment_synapse",
            "assignment_segmentation",
        ])
        prob = [prob[k] for k in keys] if prob is not None else prob
        dp.set_sampling_weights(p=prob)
        self.dataprovider = dp
        self.extras = extras
        print(dp)

    def build_dataset(
        self,
        tag: str,
        data: dict[str, np.ndarray],
        spec: dict[str, tuple[int, int, int]]
    ) -> Dataset:
        """Create a Dataset."""
        dset = Dataset(tag=tag)

        dset.add_data(key="input", data=data["input"])
        dset.add_data(key="assignment_synapse", data=data["assignment_synapse"])
        dset.add_data(
            key="assignment_segmentation", data=data["assignment_segmentation"]
        )
        dset.add_mask(
            key="assignment_synapse_mask", data=data["assignment_synapse_mask"]
        )
        dset.add_mask(
            key="assignment_locations", data=data["assignment_synapse"], loc=True
        )

        extras = data["assignment_synapse_extra"]

        return dset, extras
