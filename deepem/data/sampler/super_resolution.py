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

Supports coarse loading (load_resolution != resolution) for memory-constrained
datasets. Coarse-loaded datasets are sampled at reduced resolution and upsampled
back to the training resolution before further processing.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

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

    Datasets are split into up to 4 groups based on two axes:
      - isotropy: isotropic vs anisotropic
      - load resolution: native (load_resolution == resolution) vs
        coarse (load_resolution coarser than resolution)

    Each non-empty group gets its own DataProvider with appropriately scaled
    spec. After sampling, coarse patches are upsampled, then iso/aniso
    processing is applied.

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

        # Split datasets into 4 groups: (iso/aniso) x (native/coarse)
        groups = self._split_datasets(data)

        # Build DataProviders for each non-empty group
        self.groups = {}  # name -> (DataProvider, scale_factor)
        aniso_spec = self._make_aniso_spec(spec)

        for group_name, group_data in groups.items():
            if not group_data:
                continue

            is_coarse = group_name.endswith("_coarse")
            is_iso = group_name.startswith("iso")
            base_spec = spec if is_iso else aniso_spec

            if is_coarse:
                scale = self._get_coarse_scale(group_data)
                group_spec = self._scale_spec(base_spec, scale)
            else:
                scale = (1, 1, 1)
                group_spec = base_spec

            aug_to_use = aug if is_iso else (aug_aniso or aug)

            dp = self.build_dataprovider(
                group_data, group_spec, aug_to_use, prob, zettaset_specs
            )
            self.groups[group_name] = (dp, scale)

        # Compute per-group sampling probabilities
        self.group_names, self.group_probs = self._compute_group_probs(prob)

    def _split_datasets(
        self, data: dict[str, dict[str, np.ndarray]]
    ) -> dict[str, dict]:
        """Split data into 4 groups: (iso/aniso) x (native/coarse)."""
        groups = {
            "iso_native": {},
            "iso_coarse": {},
            "aniso_native": {},
            "aniso_coarse": {},
        }

        for key, dataset_data in data.items():
            zettaset_name = key.split(':')[0] if ':' in key else key
            spec = self.zettaset_specs.get(zettaset_name, {})

            is_iso = spec.get('isotropic', False)
            is_coarse = 'load_resolution' in spec

            if is_iso and not is_coarse:
                groups["iso_native"][key] = dataset_data
            elif is_iso and is_coarse:
                groups["iso_coarse"][key] = dataset_data
            elif not is_iso and not is_coarse:
                groups["aniso_native"][key] = dataset_data
            else:
                groups["aniso_coarse"][key] = dataset_data

        return groups

    def _get_coarse_scale(
        self, group_data: dict[str, dict[str, np.ndarray]]
    ) -> tuple[float, float, float]:
        """Get the scale factor for a coarse group (load_resolution / resolution).

        All datasets in a coarse group must share the same scale factor.
        """
        scale = None
        for key in group_data:
            zettaset_name = key.split(':')[0] if ':' in key else key
            spec = self.zettaset_specs.get(zettaset_name, {})
            resolution = spec.get('resolution', [16, 16, 16])
            load_resolution = spec.get('load_resolution', resolution)
            s = tuple(l / r for l, r in zip(load_resolution, resolution))
            if scale is None:
                scale = s
            elif scale != s:
                raise ValueError(
                    f"Mixed scale factors in coarse group: {scale} vs {s}. "
                    f"All coarse datasets in the same group must share the "
                    f"same scale factor."
                )
        return scale

    def _scale_spec(
        self,
        spec: dict[str, tuple[int, int, int]],
        scale: tuple[float, float, float],
    ) -> dict[str, tuple[int, int, int]]:
        """Scale a spec by the given factors (inverse: coarser res = fewer voxels)."""
        scaled = {}
        for key, (z, y, x) in spec.items():
            scaled[key] = (
                int(z / scale[0]),
                int(y / scale[1]),
                int(x / scale[2]),
            )
        return scaled

    def _make_aniso_spec(
        self, spec: dict[str, tuple[int, int, int]]
    ) -> dict[str, tuple[int, int, int]]:
        """Derive aniso spec by dividing Z dimensions by sr_scale_z."""
        aniso_spec = {}
        for key, (z, y, x) in spec.items():
            aniso_spec[key] = (z // self.sr_scale_z, y, x)
        return aniso_spec

    def _compute_group_probs(
        self, prob: dict[str, float] | None,
    ) -> tuple[list[str], list[float]]:
        """Compute sampling probability for each non-empty group.

        Per-dataset train_prob is preserved:
          P(dataset D) = P(group G) * P(D|G) = prob_D / sum_all
        """
        group_weights = {}
        for group_name, (dp, _) in self.groups.items():
            if prob:
                w = sum(prob.get(d.tag, 1.0) for d in dp.datasets)
            else:
                # Weight by valid voxel count, scale up aniso
                w = sum(d.num_samples() for d in dp.datasets)
                if group_name.startswith("aniso"):
                    w *= self.sr_scale_z

            group_weights[group_name] = w

        total = sum(group_weights.values())
        names = list(group_weights.keys())
        probs = [group_weights[n] / total for n in names] if total > 0 else []
        return names, probs

    def __call__(self) -> dict[str, np.ndarray]:
        """Sample from one of the dataset groups."""
        # Pick a group
        idx = np.random.choice(len(self.group_names), p=self.group_probs)
        group_name = self.group_names[idx]
        dp, scale = self.groups[group_name]

        # Sample a patch
        sample = dp()

        # Upsample if coarse
        if group_name.endswith("_coarse"):
            sample = self._upsample_sample(sample, scale)

        # Apply iso/aniso processing
        if group_name.startswith("iso"):
            sample = self._process_iso_sample(sample)
            is_aniso = False
        else:
            sample = self._process_aniso_sample(sample)
            is_aniso = True

        sample = self.postprocess(sample)
        sr_scale_z = self.sr_scale_z if is_aniso else 0
        sample['_sr_scale_z'] = np.array(sr_scale_z, dtype=np.float32)
        return sample

    def _upsample_sample(
        self,
        sample: dict[str, np.ndarray],
        scale: tuple[float, float, float],
    ) -> dict[str, np.ndarray]:
        """Upsample a coarse-loaded patch to the training resolution.

        Uses trilinear (order=1) for images, nearest-neighbor (order=0) for
        labels and masks.
        """
        zoom_factors = tuple(1.0 / s for s in scale)

        for key in list(sample.keys()):
            data = sample[key]
            if key == 'input':
                order = 1  # trilinear
            else:
                order = 0  # nearest-neighbor for labels/masks

            if data.ndim == 3:
                sample[key] = ndimage.zoom(data, zoom_factors, order=order)
            elif data.ndim == 4:
                # (C, Z, Y, X) -> zoom spatial dims only
                channel_zooms = (1,) + zoom_factors
                sample[key] = ndimage.zoom(data, channel_zooms, order=order)

        return sample

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
