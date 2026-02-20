from __future__ import annotations

from cloudvolume import CloudVolume, Bbox
from cloudvolume.exceptions import InfoUnavailableError
import numpy as np
from numpy.typing import ArrayLike
import re

from zettasets.dataset import Dataset as Zettaset
from zettasets.sample import Sample


def parse_target_combination(target_spec: str) -> list[str]:
    """
    Parse a target specification that may contain combinations.

    Examples:
        "mye" -> ["mye"]
        "mye + ecs" -> ["mye", "ecs"]
        "a+b+c" -> ["a", "b", "c"]

    Args:
        target_spec: Target specification string

    Returns:
        List of individual target keys
    """
    # Split by '+' and strip whitespace
    targets = [t.strip() for t in target_spec.split('+')]
    return targets


def load_data(
    zettaset_paths: list[str],
    data_ids: list[str] | None = None,
    zettaset_resolution: tuple[float, float, float] | None = None,
    **kwargs
) -> dict[str, dict[str, np.ndarray]]:
    """ Load data from a zettaset."""
    if data_ids is None:
        return {}

    # Load zettasets.
    zettasets = []
    for zettaset_path in zettaset_paths:
        assert zettaset_path.startswith("gs://")
        print(f"Zettaset [{zettaset_path}]")
        zettasets.append(Zettaset(zettaset_path, "", zettaset_resolution))

    # Load data from a zettaset.
    data = {}
    for data_id in data_ids:
        for zettaset in zettasets:
            if data_id in zettaset.sample_names:
                print(f"Sample [{data_id}]")
                data[data_id] = load_sample(
                    zettaset.samples[data_id],
                    zettaset_resolution,
                    **kwargs
                )
                break
        if data_id not in data:
            raise KeyError(f"Invalid data id:{data_id}")

    return data


def load_sample(
    sample: Sample,
    zettaset_resolution: tuple[float, float, float] | None = None,
    zettaset_lookup: dict[str, str] | None = None,
    zettaset_padding: tuple[int, int, int] = (0, 0, 0),
    zettaset_mask: bool = True,
    requires_binarize: list[str] = [],
    **kwargs
) -> dict[str, np.ndarray]:
    """Load image and labels from a Sample."""

    def convert_array(arr: ArrayLike) -> np.ndarray:
        return np.array(arr).transpose(3,2,1,0)[0, ...]

    dset: dict[str, np.ndarray] = {}

    # Bbox with padding
    resolution = zettaset_resolution or sample.base_resolution
    bbox = sample.bbox * (sample.base_resolution / np.array(resolution))
    xyz_padding = (
        tuple(reversed(zettaset_padding))
        if zettaset_padding != (0, 0, 0)
        else (0, 0, 0)
    )
    image_bbox = Bbox(bbox.minpt - xyz_padding, bbox.maxpt + xyz_padding)

    # Image: try src_image_path first for backward compatibility,
    #        fall back to sample's own image volume
    try:
        if sample.src_image_path is None:
            raise ValueError("No src_image_path available")
        vol = CloudVolume(  # pylint: disable=unsubscriptable-object
            sample.src_image_path,
            mip=resolution,
            fill_missing=True,
            bounded=False,
        )[image_bbox.to_slices()]
    except (InfoUnavailableError, ValueError) as e:
        print(f"\tWarning: src_image_path failed ({e}), falling back to sample image volume")
        vol = CloudVolume(  # pylint: disable=unsubscriptable-object
            sample.volumes["image"].cloudpath,
            mip=resolution,
            fill_missing=True,
            bounded=False,
        )[image_bbox.to_slices()]
    dset["input"] = convert_array(vol)
    dset["input"] = (dset["input"] / 255.).astype('float32')
    print(f"input: {dset['input'].shape}")

    # Assumes that zettaset's annotation names follow DeepEM's convention.
    if zettaset_lookup is None:
        zettaset_lookup = {x: x for x in sample.annotation_names}

    # Annotations
    for name, key_spec in zettaset_lookup.items():

        # Parse target combination (e.g., "mye + ecs" -> ["mye", "ecs"])
        target_keys = parse_target_combination(key_spec)

        # Load and combine targets
        combined_data = None
        combined_mask = None

        for key in target_keys:
            # Annotation
            vol = sample.read(key)[key]
            data_array = convert_array(vol)

            # Binarize if needed (before combining)
            if name in requires_binarize:
                data_array = (data_array > 0).astype('uint8')

            # Combine (logical OR for binary targets)
            if combined_data is None:
                combined_data = data_array
            else:
                combined_data = np.maximum(combined_data, data_array)

            # Mask
            if zettaset_mask and (key in sample.masks):
                vol = sample.read_mask(key)[key]
                mask_array = convert_array(vol).astype('uint8')
            else:
                mask_array = np.ones_like(data_array, dtype='uint8')

            # Combine masks (logical OR)
            if combined_mask is None:
                combined_mask = mask_array
            else:
                combined_mask = np.maximum(combined_mask, mask_array)

        dset[name] = combined_data
        print(f"{name}: {dset[name].shape} (combined from {target_keys})")

        dset[name + "_mask"] = combined_mask
        print(f"{name + '_mask'}: {dset[name + '_mask'].shape}")

        # applying padding
        if zettaset_padding != (0, 0, 0):
            widths = (
                (zettaset_padding[0], zettaset_padding[0]),
                (zettaset_padding[1], zettaset_padding[1]),
                (zettaset_padding[2], zettaset_padding[2]),
            )

            dset[name] = np.pad(dset[name], widths, "constant", constant_values=0)
            print(f"{name} padded to {dset[name].shape}")
            dset[f"{name}_mask"] = np.pad(
                dset[f"{name}_mask"], widths, "constant", constant_values=0
            )
            print(f"{name}_mask padded to {dset[name + '_mask'].shape}")

    return dset


if __name__ == "__main__":
    ZETTASET_PATH = "gs://zetta-research-nickt-volumes/wktools_testing/dataset3"
    dataset = load_data(
        ZETTASET_PATH,
        data_ids=["6388f1170100009d0023f9ef", "63924770010000200023faf3"],
        zettaset_lookup={"soma": "somas"}
    )
    print(dataset)
