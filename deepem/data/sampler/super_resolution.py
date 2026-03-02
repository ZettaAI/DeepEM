"""
Super-resolution sampler for mixed isotropic/anisotropic training.

This sampler handles two data types with completely separate pipelines:
- Isotropic data: iso-safe augmentation → CubicSubsampleZ (input only)
  The Sampler then creates aniso labels by subsampling iso labels.
- Anisotropic data: aniso augmentation (misalign, missing, etc.)
  Labels are renamed to *_aniso, iso labels set to None.
"""
from __future__ import annotations

import numpy as np

from augmentor import Augment
from augmentor import utils as aug_utils
from dataprovider3 import DataProvider, Dataset, DataSuperset

from deepem.data.sampler.zettaset import Sampler as BaseSampler


def get_spec(
    in_spec: dict[str, tuple[int, ...]],
    out_spec: dict[str, tuple[int, ...]],
    sr_mode: bool = False,
    sr_scale_z: int = 5,
) -> dict[str, tuple[int, int, int]]:
    """
    Get the data specification for super-resolution training.

    For SR mode, output key Z dimensions are converted to aniso resolution
    (divided by sr_scale_z). CubicSubsampleZ.prepare() then expands back
    to cubic for the iso pipeline, while the aniso pipeline uses these
    dimensions directly.
    """
    spec = {}

    for key, dims in {**in_spec, **out_spec}.items():
        spatial_dims = dims[-3:]

        # Convert output keys to aniso Z dimensions for SR mode
        if sr_mode and key in out_spec:
            z, y, x = spatial_dims
            spatial_dims = (z // sr_scale_z, y, x)

        spec[key] = spatial_dims
        if key in out_spec:
            spec[f"{key}_mask"] = spatial_dims

    return spec


class Sampler(BaseSampler):
    """
    Sampler for super-resolution training with mixed iso/aniso data.

    For iso samples:
    - The augmentation pipeline produces input at aniso resolution and
      labels at iso resolution (CubicSubsampleZ subsamples input only).
    - The Sampler creates aniso labels by subsampling iso labels.
    - Output: input, key (iso), key_aniso, key_mask (iso), key_mask_aniso

    For aniso samples:
    - Standard aniso augmentation, all at aniso resolution.
    - Labels renamed to *_aniso, iso keys set to None.
    - Output: input, key=None, key_aniso, key_mask=None, key_mask_aniso
    """
    def __init__(
        self,
        data: dict[str, dict[str, np.ndarray]],
        spec: dict[str, tuple[int, int, int]],
        is_train: bool,
        aug: Augment | None = None,
        aug_aniso: Augment | None = None,
        prob: dict[str, float] | None = None,
        zettaset_specs: dict[str, dict] | None = None,
        sr_mode: bool = False,
        sr_scale_z: int = 5,
        out_spec: dict[str, tuple[int, ...]] | None = None,
        **kwargs,
    ):
        self.is_train = is_train
        self.sr_mode = sr_mode
        self.sr_scale_z = sr_scale_z
        self.out_spec = out_spec or {}
        self.zettaset_specs = zettaset_specs or {}

        # Split datasets by isotropy
        iso_data, aniso_data = self._split_by_isotropy(data)

        self.has_iso = len(iso_data) > 0
        self.has_aniso = len(aniso_data) > 0

        # Create dataproviders with separate augmentation pipelines
        if self.has_iso:
            self.dataprovider_iso = self.build_dataprovider(
                iso_data, spec, aug, prob, zettaset_specs
            )

        if self.has_aniso:
            self.dataprovider_aniso = self.build_dataprovider(
                aniso_data, spec, aug_aniso or aug, prob, zettaset_specs
            )

        # Compute sampling weights between iso and aniso
        self.iso_prob = self._compute_iso_prob(iso_data, aniso_data, prob)

    def _split_by_isotropy(
        self, data: dict[str, dict[str, np.ndarray]]
    ) -> tuple[dict, dict]:
        """Split data into isotropic and anisotropic datasets."""
        iso_data = {}
        aniso_data = {}

        for key, dataset_data in data.items():
            zettaset_name = key.split(':')[0] if ':' in key else key
            spec = self.zettaset_specs.get(zettaset_name, {})
            is_isotropic = spec.get('isotropic', False)

            if is_isotropic:
                iso_data[key] = dataset_data
            else:
                aniso_data[key] = dataset_data

        return iso_data, aniso_data

    def _compute_iso_prob(
        self,
        iso_data: dict,
        aniso_data: dict,
        prob: dict[str, float] | None,
    ) -> float:
        """Compute probability of sampling isotropic data."""
        if not self.has_iso:
            return 0.0
        if not self.has_aniso:
            return 1.0

        if prob:
            iso_total = sum(prob.get(k, 1.0) for k in iso_data)
            aniso_total = sum(prob.get(k, 1.0) for k in aniso_data)
            total = iso_total + aniso_total
            return iso_total / total if total > 0 else 0.5

        n_iso = len(iso_data)
        n_aniso = len(aniso_data)
        return n_iso / (n_iso + n_aniso)

    def __call__(self) -> dict[str, np.ndarray]:
        """Sample from either isotropic or anisotropic data."""
        sample_iso = np.random.rand() < self.iso_prob

        if sample_iso and self.has_iso:
            sample = self.dataprovider_iso()
            sample = self._process_iso_sample(sample)
        elif self.has_aniso:
            sample = self.dataprovider_aniso()
            sample = self._process_aniso_sample(sample)
        else:
            sample = self.dataprovider_iso()
            sample = self._process_iso_sample(sample)

        return self.postprocess(sample)

    def _process_iso_sample(
        self, sample: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """
        Process isotropic sample for SR training.

        After iso augmentation + CubicSubsampleZ:
        - input is at aniso resolution (subsampled)
        - labels/masks are at iso resolution (cropped, not subsampled)

        This method creates aniso labels by subsampling the iso labels.
        """
        processed = {}

        # Input stays as-is (already at aniso resolution)
        if 'input' in sample:
            processed['input'] = sample['input']

        # Process output keys: keep iso labels, create aniso by subsampling
        for key in self.out_spec:
            if key in sample:
                processed[key] = sample[key]
                processed[f'{key}_aniso'] = self._subsample_nearest(
                    sample[key], self.sr_scale_z
                )

            mask_key = f'{key}_mask'
            if mask_key in sample:
                processed[mask_key] = sample[mask_key]
                processed[f'{key}_mask_aniso'] = self._subsample_nearest(
                    sample[mask_key], self.sr_scale_z
                )

        processed['is_isotropic'] = np.array([1], dtype=np.float32)
        return processed

    def _process_aniso_sample(
        self, sample: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """
        Process anisotropic sample for SR training.

        - Rename labels to *_aniso keys
        - Set iso labels to None
        - Add is_isotropic flag (False)
        """
        processed = {}

        if 'input' in sample:
            processed['input'] = sample['input']

        for key in self.out_spec:
            if key in sample:
                processed[key] = None
                processed[f'{key}_aniso'] = sample[key]

            mask_key = f'{key}_mask'
            if mask_key in sample:
                processed[mask_key] = None
                processed[f'{key}_mask_aniso'] = sample[mask_key]

        processed['is_isotropic'] = np.array([0], dtype=np.float32)
        return processed

    @staticmethod
    def _subsample_nearest(data, factor):
        """Subsample by taking the middle slice of each block in Z."""
        offset = factor // 2
        if data.ndim == 3:
            return data[offset::factor, :, :]
        else:
            return data[:, offset::factor, :, :]

    def postprocess(
        self, sample: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Convert sample to float32 tensors, handling None and 1D arrays."""
        result = {}
        for k, v in sample.items():
            if v is None:
                result[k] = v
            elif isinstance(v, np.ndarray):
                if v.ndim >= 2:
                    v = aug_utils.to_tensor(v)
                result[k] = v.astype(np.float32)
            else:
                result[k] = v
        return result
