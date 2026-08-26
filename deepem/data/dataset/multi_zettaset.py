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

    Use "?" as the last fallback to allow zero-filling when no options match
    (instead of raising a KeyError).

    Examples:
        "seg_out" -> ["seg_out"]
        "seg_out | seg" -> ["seg_out", "seg"]
        "a|b|c" -> ["a", "b", "c"]
        "out | mye | ?" -> ["out", "mye", "?"]

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
    exclude = set(zettaset_specs.get(zettaset_name, {}).get("exclude", []))

    if exclude:
        print(f"Zettaset '{zettaset_name}' exclude: {sorted(exclude)}")
        print(f"Zettaset '{zettaset_name}' loading: "
              f"{sorted(s for s in zettaset.sample_names if s not in exclude)}")

    for sample_name in zettaset.sample_names:
        if sample_name in exclude:
            continue
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

    # Per-dataset share_mask overrides global zettaset_share_mask
    if "share_mask" in zettaset_spec:
        kwargs = {**kwargs, "zettaset_share_mask": zettaset_spec["share_mask"]}

    known_absent = zettaset_spec.get("known_absent", [])

    print(f"Sample [{data_id}]")
    return {data_id: load_sample(
        zettaset.samples[sample_name],
        padding,
        no_mask,
        resolution,
        known_absent=known_absent,
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
    known_absent: list[str] = [],
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
    # float32, not the float64 that uint8 / 255.0 would promote to: at 8 nm a
    # padded hemibrain sample is ~314 M voxels, so the dtype is worth 1.3 GB.
    dset["input"] = convert_array(vol).astype(np.float32) / np.float32(255.0)
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

    # Several lookup names routinely resolve to the same source annotation --
    # embedding / affinity / long_range all read "seg" -- and each one used to
    # read, transform and pad its own full-size copy. Materialize each distinct
    # (source, transform) once per sample and share it by reference across the
    # names that resolve to it. Safe by construction: DataProvider3's TensorData
    # is read-only and get_patch() returns np.copy(), so nothing downstream can
    # mutate these volumes. Under forked DataLoader workers it also means fewer
    # copy-on-write pages.
    raw_cache: dict[str, np.ndarray] = {}
    data_cache: dict[tuple, np.ndarray] = {}
    mask_cache: dict[tuple, np.ndarray] = {}
    cache_owner: dict[tuple, str] = {}

    def read_annotation(key: str) -> np.ndarray:
        """Read one annotation volume, at most once per sample."""
        if key not in raw_cache:
            raw_cache[key] = convert_array(sample.read(key)[key])
        return raw_cache[key]

    def build_data(target_keys: list[str], transform: tuple) -> np.ndarray:
        combined = None
        for key in target_keys:
            data_array = read_annotation(key)

            # Semantic mapping or binarize (before combining)
            if transform[0] == "sem":
                data_array = (data_array == transform[1]).astype("uint8")
            elif transform[0] == "bin":
                data_array = (data_array > 0).astype("uint8")

            # Combine (logical OR for binary targets)
            combined = (
                data_array if combined is None
                else np.maximum(combined, data_array)
            )
        return combined

    def build_mask(target_keys: list[str]) -> np.ndarray:
        combined = None
        for key in target_keys:
            if (not no_mask) and (key in sample.masks):
                # Always prioritize loading own mask if it exists
                mask_vol = sample.read_mask(key)[key]
                mask_array = convert_array(mask_vol).astype("uint8")
            elif shared_mask is not None:
                # Fall back to shared mask if no specific mask exists
                mask_array = shared_mask
            else:
                # Only the shape was ever taken from the annotation, and every
                # transform is elementwise -- so the mask does not depend on
                # which lookup name we are serving.
                mask_array = np.ones(read_annotation(key).shape, dtype="uint8")

            # Combine masks (logical OR)
            combined = (
                mask_array if combined is None
                else np.maximum(combined, mask_array)
            )
        return combined

    def pad(array: np.ndarray) -> np.ndarray:
        if padding == (0, 0, 0):
            return array
        return np.pad(array, tuple((p, p) for p in padding), "constant")

    # Process annotations
    for name, key_spec in zettaset_lookup.items():

        # Parse fallback options first (e.g., "seg_out | seg" -> ["seg_out", "seg"])
        fallback_options = parse_target_fallback(key_spec)

        # Check if zero-fill is allowed (trailing "?" in fallback options)
        allow_zero_fill = fallback_options[-1] == "?"
        if allow_zero_fill:
            fallback_options = fallback_options[:-1]

        # Try each fallback option until one exists
        selected_key_spec = None
        for option in fallback_options:
            # Parse target combination for this option (e.g., "mye + ecs" -> ["mye", "ecs"])
            target_keys = parse_target_combination(option)
            # Check if all targets in this combination exist
            if all(key in sample.annotation_names for key in target_keys):
                selected_key_spec = option
                break

        if selected_key_spec is None and allow_zero_fill:
            # Shape derived from bbox (ZYX order, matching convert_array output).
            bbox_size = bbox.maxpt - bbox.minpt
            shape = tuple(int(s) for s in reversed(bbox_size))
            target_keys = []
            # Check if annotation is known to be absent (negative example).
            is_negative = any(key in known_absent for key in fallback_options)
            data_cache_key = ("__zerofill__",)
            mask_cache_key = ("__zerofill__", is_negative)
            if data_cache_key not in data_cache:
                data_cache[data_cache_key] = pad(np.zeros(shape, dtype="float32"))
            if mask_cache_key not in mask_cache:
                fill = np.ones if is_negative else np.zeros
                mask_cache[mask_cache_key] = pad(fill(shape, dtype="uint8"))
            if is_negative:
                print(
                    f"\t'{name}' - none of the fallback options "
                    f"{fallback_options} found in annotations "
                    f"{sample.annotation_names}. "
                    f"Known absent: using all-ones mask (negative example)."
                )
            else:
                print(
                    f"\tWARNING: '{name}' - none of the fallback options "
                    f"{fallback_options} found in annotations "
                    f"{sample.annotation_names}. "
                    f"Zero-filling with all-zero mask (no loss contribution)."
                )
        elif selected_key_spec is None:
            raise KeyError(
                f"None of the fallback options {fallback_options} exist in "
                f"sample annotations. "
                f"Available annotations: {sample.annotation_names}. "
                f"Use '?' as trailing fallback to allow zero-filling "
                f"(e.g., \"{key_spec} | ?\")."
            )
        else:
            # Parse target combination (e.g., "mye + ecs" -> ["mye", "ecs"])
            target_keys = parse_target_combination(selected_key_spec)

            # The transform is the only per-name variation in the data branch.
            if name in semantic_mapping:
                transform = ("sem", semantic_mapping[name])
            elif name in requires_binarize:
                transform = ("bin",)
            else:
                transform = ("raw",)

            # Key on the resolved targets, not the raw spec string: fallback can
            # pick a different option for each name, and "mye + ecs" has to
            # normalize to the same tuple. The mask needs no transform component
            # -- it is a pure function of the source keys.
            data_cache_key = (tuple(target_keys), transform)
            mask_cache_key = (tuple(target_keys),)
            # Consulted independently: a name can hit the mask cache and miss
            # the data cache (same source, different transform).
            if data_cache_key not in data_cache:
                data_cache[data_cache_key] = pad(build_data(target_keys, transform))
            if mask_cache_key not in mask_cache:
                mask_cache[mask_cache_key] = pad(build_mask(target_keys))

        mask_key = f"{name}_mask"
        dset[name] = data_cache[data_cache_key]
        dset[mask_key] = mask_cache[mask_cache_key]

        owner = cache_owner.setdefault(data_cache_key, name)
        anno_log = f"\t{name}: {dset[name].shape} (combined from {target_keys})"
        if owner != name:
            anno_log += f" -> sharing array with '{owner}'"
        print(anno_log)
        print(f"\t{mask_key}: {dset[mask_key].shape}")

    # The unpadded reads are only needed while the caches above are filled.
    raw_cache.clear()

    return dset


if __name__ == "__main__":
    pass
