"""
Super-resolution sampler for mixed isotropic/anisotropic training.

Uses zero-padding so a standard isotropic RSUNet always sees iso-shaped tensors.
Real sections are placed at offset::sr_scale_z positions; padded positions are
zeros. The zero-padded masks naturally cause the loss to ignore padded positions,
and z-affinity is automatically masked out because generate_mask_aff multiplies
neighbor masks (no adjacent real sections exist in aniso data).

- Isotropic data: avg-downsample input in Z then zero-pad (emulates aniso
  input), labels/masks stay at full iso resolution for full supervision.
- Anisotropic data: zero-pad ALL keys (input, targets, masks) in Z.
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

    Returns standard isotropic spec. The sampler internally derives aniso
    spec for the aniso dataprovider.
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

    Uses zero-padding so the model always sees iso-shaped tensors.

    For iso samples:
    - Input is avg-downsampled in Z then zero-padded back to iso size
    - Labels/masks stay at full iso resolution (full supervision)

    For aniso samples:
    - ALL keys (input, targets, masks) are zero-padded in Z to iso size
    """
    def __init__(
        self,
        data: dict[str, dict[str, np.ndarray]],
        spec: dict[str, tuple[int, int, int]],
        is_train: bool,
        aug: Augment | None = None,
        prob: dict[str, float] | None = None,
        zettaset_specs: dict[str, dict] | None = None,
        aug_aniso: Augment | None = None,
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

        # Iso dataprovider uses iso spec directly
        if self.has_iso:
            self.dataprovider_iso = self.build_dataprovider(
                iso_data, spec, aug, prob, zettaset_specs
            )

        # Aniso dataprovider uses aniso spec (Z / sr_scale_z)
        if self.has_aniso:
            aniso_spec = self._make_aniso_spec(spec)
            self.dataprovider_aniso = self.build_dataprovider(
                aniso_data, aniso_spec, aug_aniso or aug, prob, zettaset_specs
            )

        # Compute sampling weights between iso and aniso
        self.iso_prob = self._compute_iso_prob(prob)

    def _make_aniso_spec(
        self, spec: dict[str, tuple[int, int, int]]
    ) -> dict[str, tuple[int, int, int]]:
        """Derive aniso spec by dividing Z dimensions by sr_scale_z."""
        aniso_spec = {}
        for key, (z, y, x) in spec.items():
            aniso_spec[key] = (z // self.sr_scale_z, y, x)
        return aniso_spec

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
        prob: dict[str, float] | None,
    ) -> float:
        """Compute probability of sampling isotropic data.

        Uses valid voxel counts from each dataprovider. Aniso voxel count
        is scaled by sr_scale_z to compensate for the smaller Z spec.
        User-provided prob weights override voxel-count-based weighting.
        """
        if not self.has_iso:
            return 0.0
        if not self.has_aniso:
            return 1.0

        if prob:
            # Use user-provided weights (already set on each dataprovider)
            iso_total = sum(
                prob.get(k, 1.0)
                for dp in [self.dataprovider_iso]
                for k in [d.tag for d in dp.datasets]
            )
            aniso_total = sum(
                prob.get(k, 1.0)
                for dp in [self.dataprovider_aniso]
                for k in [d.tag for d in dp.datasets]
            )
        else:
            # Weight by valid voxel count
            iso_total = sum(
                d.num_samples() for d in self.dataprovider_iso.datasets
            )
            # Aniso spec has Z/sr_scale_z, so scale up to match iso
            aniso_total = sum(
                d.num_samples() for d in self.dataprovider_aniso.datasets
            ) * self.sr_scale_z

        total = iso_total + aniso_total
        return iso_total / total if total > 0 else 0.5

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

        Avg-downsample input in Z then zero-pad back to iso size (emulates
        zero-padded aniso input). Labels/masks stay at full iso resolution.
        """
        for key in list(sample.keys()):
            if key == 'input':
                sample[key] = _avg_downsample_and_zero_pad_z(
                    sample[key], self.sr_scale_z
                )
            # Labels and masks stay at full iso resolution (no modification)
        return sample

    def _process_aniso_sample(
        self, sample: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """
        Process anisotropic sample for SR training.

        Zero-pad ALL keys (input, targets, masks) in Z to iso size.
        """
        for key in list(sample.keys()):
            sample[key] = _zero_pad_z(sample[key], self.sr_scale_z)
        return sample


def _zero_pad_z(data: np.ndarray, sr_scale_z: int) -> np.ndarray:
    """
    Zero-pad array in Z dimension.

    Places real sections at offset::sr_scale_z where offset = sr_scale_z // 2.

    Args:
        data: 3D (Z, Y, X) or 4D (C, Z, Y, X) array
        sr_scale_z: Upsampling factor

    Returns:
        Zero-padded array with Z dimension multiplied by sr_scale_z
    """
    offset = sr_scale_z // 2

    if data.ndim == 3:
        z, y, x = data.shape
        padded = np.zeros((z * sr_scale_z, y, x), dtype=data.dtype)
        padded[offset::sr_scale_z, :, :] = data
    elif data.ndim == 4:
        c, z, y, x = data.shape
        padded = np.zeros((c, z * sr_scale_z, y, x), dtype=data.dtype)
        padded[:, offset::sr_scale_z, :, :] = data
    else:
        raise ValueError(f"Expected 3D or 4D array, got {data.ndim}D")

    return padded


def _avg_downsample_and_zero_pad_z(
    data: np.ndarray, sr_scale_z: int
) -> np.ndarray:
    """
    Average-downsample in Z by sr_scale_z, then zero-pad back to original Z size.

    Applied to iso input only. Degrades iso input to look like zero-padded
    aniso while labels remain at full iso resolution for full supervision.

    Args:
        data: 3D (Z, Y, X) or 4D (C, Z, Y, X) array
        sr_scale_z: Downsampling/upsampling factor

    Returns:
        Zero-padded array with same Z size as input
    """
    if data.ndim == 3:
        z, y, x = data.shape
        # Reshape to (z//factor, factor, y, x) and average
        downsampled = data.reshape(z // sr_scale_z, sr_scale_z, y, x).mean(axis=1)
    elif data.ndim == 4:
        c, z, y, x = data.shape
        # Reshape to (c, z//factor, factor, y, x) and average
        downsampled = data.reshape(c, z // sr_scale_z, sr_scale_z, y, x).mean(axis=2)
    else:
        raise ValueError(f"Expected 3D or 4D array, got {data.ndim}D")

    return _zero_pad_z(downsampled, sr_scale_z)
