from __future__ import annotations

from cloudvolume import CloudVolume, Bbox
import numpy as np
from numpy.typing import ArrayLike

from zettasets.dataset import Dataset as Zettaset
from zettasets.sample import Sample


def load_data(
    zettaset_path: str,
    data_ids: list[str] | None = None,
    **kwargs
) -> dict[str, dict[str, np.ndarray]]:
    """ Load data from a zettaset."""
    if data_ids is None:
        return {}

    # Load a zettaset.
    assert zettaset_path.startswith("gs://")
    motivation = "Load a zettaset from DeepEM"
    zettaset = Zettaset(zettaset_path, motivation)

    # Load data from a zettaset.
    data = {}
    for data_id in data_ids:
        if data_id in zettaset.sample_names:
            print(f"Sample [{data_id}]")
            data[data_id] = load_sample(zettaset.samples[data_id], **kwargs)
        else:
            raise KeyError(f"Invalid data id:{data_id}")

    return data


def load_sample(
    sample: Sample,
    zettaset_lookup: dict[str, str] | None = None,
    zettaset_padding: tuple[int, int, int] = (0, 0, 0),
    requires_binarize: list[str] = [],
    **kwargs
) -> dict[str, np.ndarray]:
    """Load image and labels from a Sample."""

    def convert_array(arr: ArrayLike) -> np.ndarray:
        return np.array(arr).transpose(3,2,1,0)[0, ...]

    dset: dict[str, np.ndarray] = {}

    # Image
    if zettaset_padding == (0, 0, 0):
        image_bbox = sample.bbox
    else:
        xyz_padding = tuple(reversed(zettaset_padding))
        image_bbox = Bbox(
            sample.bbox.minpt - xyz_padding, sample.bbox.maxpt + xyz_padding
        )

    vol = CloudVolume(  # pylint: disable=unsubscriptable-object
        sample.src_image_path,
        mip=sample.base_resolution,
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
    for name, key in zettaset_lookup.items():

        # Annotation
        vol = sample.read(key)[key]
        dset[name] = convert_array(vol)
        print(f"{name}: {dset[name].shape}")

        # Binarize
        if name in requires_binarize:
            dset[name] = (dset[name] > 0).astype('uint8')

        # Mask
        if key in sample.masks:
            vol = sample.read_mask(key)[key]
            dset[name + "_mask"] = convert_array(vol).astype('uint8')
        else:
            dset[name + "_mask"] = np.ones_like(dset[name], dtype='uint8')
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
