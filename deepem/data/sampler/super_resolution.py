"""
Super-resolution sampler for mixed isotropic/anisotropic training.

This sampler handles two data types:
- Isotropic data: Augmentation includes CubicSubsampleZ which produces both
  iso and aniso labels automatically
- Anisotropic data: Standard aniso augmentation, labels renamed to *_aniso

The augmentation pipeline handles:
- Iso: FlipRotateIsotropic → CubicSubsampleZ → shared aniso augmentation
- Aniso: shared aniso augmentation only
"""
from __future__ import annotations

import numpy as np

from augmentor import Augment
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

    Note: The actual spec expansion for iso data (cubic) is handled by
    CubicSubsampleZ.prepare() in the augmentation pipeline.

    This returns the target aniso spec that CubicSubsampleZ will crop/subsample to.
    """
    spec = {}

    for key, dims in {**in_spec, **out_spec}.items():
        spatial_dims = dims[-3:]
        spec[key] = spatial_dims
        if key in out_spec:
            spec[f"{key}_mask"] = spatial_dims

    return spec


class Sampler(BaseSampler):
    """
    Sampler for super-resolution training with mixed iso/aniso data.

    The augmentation pipelines handle the complexity:
    - Iso augmentation: CubicSubsampleZ produces both iso and aniso labels
    - Aniso augmentation: Labels are renamed to *_aniso in postprocess
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

        # Create dataproviders with appropriate augmentation
        # Iso augmentation includes CubicSubsampleZ which expands spec internally
        if self.has_iso:
            self.dataprovider_iso = self.build_dataprovider(
                iso_data, spec, aug, prob, zettaset_specs
            )

        # Aniso augmentation uses native aniso spec
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
            # Check zettaset_specs for isotropic flag
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

        # Use provided probabilities if available
        if prob:
            iso_total = sum(prob.get(k, 1.0) for k in iso_data)
            aniso_total = sum(prob.get(k, 1.0) for k in aniso_data)
            total = iso_total + aniso_total
            return iso_total / total if total > 0 else 0.5

        # Default: equal weight per dataset
        n_iso = len(iso_data)
        n_aniso = len(aniso_data)
        return n_iso / (n_iso + n_aniso)

    def __call__(self) -> dict[str, np.ndarray]:
        """Sample from either isotropic or anisotropic data."""
        # Decide whether to sample iso or aniso
        sample_iso = np.random.rand() < self.iso_prob

        if sample_iso and self.has_iso:
            # Iso sample: augmentation already produced iso + aniso labels
            sample = self.dataprovider_iso()
        elif self.has_aniso:
            # Aniso sample: need to add is_isotropic flag and rename labels
            sample = self.dataprovider_aniso()
            sample = self._process_aniso_sample(sample)
        else:
            # Fallback to iso if no aniso data
            sample = self.dataprovider_iso()

        return self.postprocess(sample)

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

        # Input stays as-is
        if 'input' in sample:
            processed['input'] = sample['input']

        # Process output keys
        for key in self.out_spec:
            if key in sample:
                # No iso target for aniso data
                processed[key] = None
                # Aniso target
                processed[f'{key}_aniso'] = sample[key]

            # Process masks
            mask_key = f'{key}_mask'
            if mask_key in sample:
                processed[mask_key] = None
                processed[f'{key}_mask_aniso'] = sample[mask_key]

        # Mark as anisotropic
        processed['is_isotropic'] = np.array([0], dtype=np.float32)

        return processed

    def postprocess(
        self, sample: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Convert sample to float32 tensors."""
        sample = Augment.to_tensor(sample)
        return self.convert_to_float32(sample)

    def convert_to_float32(
        self, sample: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Convert all arrays to float32, handling None values."""
        result = {}
        for k, v in sample.items():
            if v is None:
                result[k] = v
            else:
                result[k] = v.astype(np.float32)
        return result
