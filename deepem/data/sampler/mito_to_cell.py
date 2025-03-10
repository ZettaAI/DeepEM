from __future__ import annotations

import numpy as np

from augmentor import Augment

from deepem.data.sampler.zettaset import Sampler as ZettasetSampler
from deepem.utils.py_utils import crop_center


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


class Sampler(ZettasetSampler):
    def __init__(
        self,
        data: dict[str, dict[str, np.ndarray]],
        spec: dict[str, tuple[int, int, int]],
        is_train: bool,
        aug: Augment | None = None,
        prob: dict[str, float] | None = None,
        zettaset_specs: dict[str, dict] | None = None,
        mode: str = "random",
        output_shape: tuple[int, int, int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(data, spec, is_train, aug, prob, zettaset_specs)
        self.dataprovider.segs += ["input_mitochondria", "mitochondria_to_cell"]
        self.mode = mode
        self.output_shape = output_shape

    def postprocess(
        self, sample: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        if "mitochondria_to_cell" in sample:
            mito_input = sample.pop("input_mitochondria")
            mito_to_cell = sample.pop("mitochondria_to_cell")
            mito_to_cell_mask = sample.pop("mitochondria_to_cell_mask")

            mito_ids = self.pick_mitochondria(mito_input, mode=self.mode)

            sample["input_mitochondria"] = self.prepare_assignment_input(mito_input, mito_ids)
            sample["mitochondria_to_cell"] = self.prepare_assignment_output(mito_input, mito_to_cell, mito_ids)
            sample["mitochondria_to_cell_mask"] = crop_center(mito_to_cell_mask, self.output_shape)
        return super().postprocess(sample)

    def convert_to_float32(
        self, sample: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        return {k: v.astype(np.float32) for k, v in sample.items()}

    def pick_mitochondria(self, mito: np.ndarray, mode: str = "random") -> list[int]:
        mito_ids = np.unique(mito[mito > 0])

        if len(mito_ids) == 0:
            return []

        if mode == "all":
            return mito_ids.tolist()
        elif mode == "random":
            return [np.random.choice(mito_ids)]
        elif mode == "multiple":
            n = np.random.randint(1, len(mito_ids) + 1)
            return np.random.choice(mito_ids, size=n, replace=False).tolist()
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def prepare_assignment_input(
        self, mito: np.ndarray, mito_ids: list[int]
    ) -> np.ndarray:
        if mito_ids:
            mito_mask = np.isin(mito, mito_ids)
            return mito_mask.astype(np.uint8)
        return np.zeros_like(mito, dtype=np.uint8)

    def prepare_assignment_output(
        self, mito_input: np.ndarray, mito_to_cell: np.ndarray, mito_ids: list[int]
    ) -> np.ndarray:
        if not mito_ids:
            return np.zeros(self.output_shape, dtype=np.float32)

        mito_mask = np.isin(mito_input, mito_ids)
        seg_ids = np.unique(mito_to_cell[mito_mask & (mito_to_cell > 0)])
        mito_to_cell_mask = np.isin(mito_to_cell, seg_ids).astype(np.float32)
        return crop_center(mito_to_cell_mask, self.output_shape)
