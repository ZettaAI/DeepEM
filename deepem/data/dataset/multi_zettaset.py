from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
import re

from cloudvolume import CloudVolume, Bbox

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


def parse_target_fallback(target_spec: str) -> list[str]:
    """
    Parse a target specification that may contain fallback options.

    Examples:
        "seg_out" -> ["seg_out"]
        "seg_out | seg" -> ["seg_out", "seg"]
        "a|b|c" -> ["a", "b", "c"]

    Args:
        target_spec: Target specification string with | as fallback separator

    Returns:
        List of target keys in priority order (first has highest priority)
    """
    # Split by '|' and strip whitespace
    targets = [t.strip() for t in target_spec.split('|')]
    return targets


def is_valid_format(s: str) -> bool:
    """Check if string has format 'A:B'."""
    return bool(re.match(r'^[^:]+:[^:]+$', s))


def load_data(
    zettaset_specs: dict[str, dict],
    data_ids: list[str] | None = None,
    **kwargs,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Load data from a zettaset.

    Parameters:
    - zettaset_specs (dict[str, dict]): Specifications for the zettasets to load.
        Each key is the name of the zettaset, and the value is a dictionary of
        its specifications.
    - data_ids (list[str], optional): Specific data identifiers to load. Defaults to None.

    Returns:
    - dict[str, dict[str, np.ndarray]]: A dictionary containing the loaded zettaset data.

    Raises:
    - ValueError: If `zettaset_specs` is empty or contains invalid specifications.
    - KeyError: If any data id is not found in the loaded zettasets.
    """
    
    if not zettaset_specs:
        raise ValueError("No zettaset specifications provided.")

    zettasets = _initialize_zettasets(zettaset_specs, **kwargs)

    data = {}
    for data_id in data_ids or []:
        if data_id in zettasets:
            data.update(_process_zettaset(data_id, zettasets, zettaset_specs, **kwargs))
        else:
            data.update(_process_sample(data_id, zettasets, zettaset_specs, **kwargs))

    return data


def _initialize_zettasets(
    zettaset_specs,
    zettaset_resolution: tuple[float, float, float] | None = None,
    **kwargs,
) -> dict[str, Zettaset]:
    zettasets = {}
    for name, spec in zettaset_specs.items():
        if "path" not in spec or not spec["path"].startswith("gs://"):
            raise ValueError(f"Invalid zettaset specification for '{name}': missing or invalid 'path'.")
        zettaset_path = spec["path"]
        print(f"Zettaset `{name}` from [{zettaset_path}]")
        resolution = tuple(spec.get("resolution", zettaset_resolution))
        print(f"{resolution=}")
        zettasets[name] = Zettaset(zettaset_path, "", resolution)
    return zettasets


def _process_zettaset(
    zettaset_name: str,
    zettasets: dict[str, Zettaset],
    zettaset_specs: dict[str, dict],
    **kwargs,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Processes a whole zettaset."""
    if zettaset_name not in zettasets:
        raise KeyError(f"Zettaset '{zettaset_name}' not found.")

    data = {}
    zettaset = zettasets[zettaset_name]

    for sample_name in zettaset.sample_names:
        full_name = f"{zettaset_name}:{sample_name}"
        data.update(_process_sample(full_name, zettasets, zettaset_specs, **kwargs))

    return {zettaset_name: data} if data else {}


def _process_sample(
    data_id: str,
    zettasets: dict[str, Zettaset],
    zettaset_specs: dict[str, dict],
    zettaset_padding: tuple[int, int, int] = (0, 0, 0),
    zettaset_padding_spec: dict[str, tuple[int, int, int]] = {},
    zettaset_mask: bool = True,
    zettaset_resolution: tuple[int, int, int] | None = None,
    **kwargs,
) -> dict[str, dict[str, np.ndarray]]:
    if not is_valid_format(data_id):
        raise ValueError(f"Invalid format for data id '{data_id}'. Expected format 'A:B'.")

    zettaset_name, sample_name = data_id.split(':')

    if zettaset_name not in zettasets:
        raise KeyError(f"Zettaset '{zettaset_name}' not found.")

    zettaset = zettasets[zettaset_name]

    if sample_name not in zettaset.sample_names:
        raise KeyError(f"Sample name '{sample_name}' not found in zettaset '{zettaset_name}'.")

    assert zettaset_name in zettaset_specs
    zettaset_spec = zettaset_specs[zettaset_name]

    # Determine padding: sample-specific overrides zettaset-specific
    padding = tuple(zettaset_padding_spec.get(data_id, zettaset_spec.get("padding", zettaset_padding)))
    no_mask = zettaset_spec.get("no_mask", not zettaset_mask)

    # Determine resolution: zettaset-specific overrides zettaset_resolution
    resolution = tuple(zettaset_spec.get("resolution", zettaset_resolution))

    print(f"Sample [{data_id}]")
    return {data_id: load_sample(
        zettaset.samples[sample_name],
        padding,
        no_mask,
        resolution,
        **kwargs,
    )}


def load_sample(
    sample: Sample,
    padding: tuple[int, int, int] = (0, 0, 0),
    no_mask: bool = False,
    resolution: tuple[int, int, int] | None = None,
    zettaset_lookup: dict[str, str] | None = None,
    requires_binarize: list[str] = [],
    zettaset_share_mask: str | None = None,
    semantic_mapping: dict[str, int] = {},
    **kwargs
) -> dict[str, np.ndarray]:
    """Load image and labels from a Sample."""

    def convert_array(arr: ArrayLike) -> np.ndarray:
        return np.array(arr).transpose(3, 2, 1, 0)[0, ...]

    dset: dict[str, np.ndarray] = {}

    # Bbox with padding
    resolution = resolution or sample.base_resolution
    bbox = sample.bbox * (sample.base_resolution / np.array(resolution))
    xyz_padding = (
        tuple(reversed(padding))
        if padding != (0, 0, 0)
        else (0, 0, 0)
    )
    image_bbox = Bbox(bbox.minpt - xyz_padding, bbox.maxpt + xyz_padding)
    
    # Image
    vol = CloudVolume(  # pylint: disable=unsubscriptable-object
        sample.src_image_path,
        mip=resolution,
        fill_missing=True,
        bounded=False,
    )[image_bbox.to_slices()]
    dset["input"] = convert_array(vol) / 255.0
    print(f"\tinput: {dset['input'].shape}")

    # Assumes that zettaset's annotation names follow DeepEM's convention.
    zettaset_lookup = zettaset_lookup or {x: x for x in sample.annotation_names}

    # Shared mask
    shared_mask = None
    if (not no_mask) and zettaset_share_mask:
        key = zettaset_share_mask
        mask_key = f"{zettaset_share_mask}_mask"
        if key not in sample.masks:
            raise KeyError(f"Mask '{mask_key}' not found.")
        mask_vol = sample.read_mask(key)[key]
        shared_mask = convert_array(mask_vol).astype("uint8")

    # Process annotations
    for name, key_spec in zettaset_lookup.items():

        # Parse fallback options first (e.g., "seg_out | seg" -> ["seg_out", "seg"])
        fallback_options = parse_target_fallback(key_spec)

        # Try each fallback option until one exists
        selected_key_spec = None
        for option in fallback_options:
            # Parse target combination for this option (e.g., "mye + ecs" -> ["mye", "ecs"])
            target_keys = parse_target_combination(option)
            # Check if all targets in this combination exist
            if all(key in sample.annotation_names for key in target_keys):
                selected_key_spec = option
                break

        if selected_key_spec is None:
            raise KeyError(
                f"None of the fallback options {fallback_options} exist in sample annotations. "
                f"Available annotations: {sample.annotation_names}"
            )

        # Parse target combination (e.g., "mye + ecs" -> ["mye", "ecs"])
        target_keys = parse_target_combination(selected_key_spec)

        # Load and combine targets
        combined_data = None
        combined_mask = None

        for key in target_keys:
            # Annotation
            vol = sample.read(key)[key]
            data_array = convert_array(vol)

            # Semantic mapping or binarize (before combining)
            if name in semantic_mapping:
                data_array = (data_array == semantic_mapping[name]).astype("uint8")
            elif name in requires_binarize:
                data_array = (data_array > 0).astype("uint8")

            # Combine (logical OR for binary targets)
            if combined_data is None:
                combined_data = data_array
            else:
                combined_data = np.maximum(combined_data, data_array)

            # Mask
            mask_key = f"{name}_mask"
            if (not no_mask) and (key in sample.masks):
                # Always prioritize loading own mask if it exists
                mask_vol = sample.read_mask(key)[key]
                mask_array = convert_array(mask_vol).astype("uint8")
            elif shared_mask is not None:
                # Fall back to shared mask if no specific mask exists
                mask_array = shared_mask
            else:
                mask_array = np.ones_like(data_array, dtype="uint8")

            # Combine masks (logical OR)
            if combined_mask is None:
                combined_mask = mask_array
            else:
                combined_mask = np.maximum(combined_mask, mask_array)

        dset[name] = combined_data
        anno_log = f"\t{name}: {dset[name].shape} (combined from {target_keys})"

        mask_key = f"{name}_mask"
        dset[mask_key] = combined_mask
        msk_log = f"\t{mask_key}: {dset[mask_key].shape}"

        # Padding
        if padding != (0, 0, 0):
            pad_width = tuple((p, p) for p in padding)
            dset[name] = np.pad(dset[name], pad_width, "constant")
            anno_log += f" -> {dset[name].shape}"
            dset[mask_key] = np.pad(dset[mask_key], pad_width, "constant")
            msk_log += f" -> {dset[mask_key].shape}"

        print(anno_log)
        print(msk_log)

    return dset


if __name__ == "__main__":
    pass
