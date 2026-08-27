"""Tests for load_sample's per-sample array deduplication.

Several zettaset_lookup names routinely resolve to the same source annotation
(embedding / affinity / long_range all read "seg"). load_sample must materialize
each distinct (source, transform) once and share it by reference, while keeping
names that need a *different* transform on separate arrays.
"""
from unittest import mock

import numpy as np

from deepem.data.dataset import multi_zettaset


SHAPE = (6, 5, 4)          # z, y, x -- small and non-cubic to catch axis bugs
PADDING = (2, 2, 2)
PADDED = tuple(d + 2 * p for d, p in zip(SHAPE, PADDING))


def _cv_order(arr):
    """numpy (z,y,x) -> the (x,y,z,c) layout convert_array() undoes."""
    return arr.transpose(2, 1, 0)[..., np.newaxis]


class FakeSample:
    """Minimal stand-in for zettasets.sample.Sample."""

    def __init__(self, annotations, masks=None):
        self._annotations = annotations
        self._masks = masks or {}
        self.base_resolution = (8, 8, 8)
        self.src_image_path = "gs://fake/image"
        self.reads = []                      # records every read for read-count assertions

    # -- the surface load_sample touches --
    @property
    def bbox(self):
        from cloudvolume import Bbox
        return Bbox((0, 0, 0), (SHAPE[2], SHAPE[1], SHAPE[0]))

    @property
    def annotation_names(self):
        return list(self._annotations)

    @property
    def masks(self):
        return self._masks

    def read(self, key):
        self.reads.append(key)
        return {key: _cv_order(self._annotations[key])}

    def read_mask(self, key):
        return {key: _cv_order(self._masks[key])}


def _load(sample, lookup, **kwargs):
    image = np.zeros(tuple(d + 2 * p for d, p in zip(SHAPE, PADDING)), dtype="uint8")
    with mock.patch.object(multi_zettaset, "CloudVolume") as cv:
        cv.return_value.__getitem__.return_value = _cv_order(image)
        return multi_zettaset.load_sample(
            sample,
            padding=PADDING,
            resolution=(8, 8, 8),
            zettaset_lookup=lookup,
            **kwargs,
        )


def _seg():
    return np.arange(int(np.prod(SHAPE)), dtype="uint32").reshape(SHAPE)


def test_same_source_same_transform_is_shared():
    seg = _seg()
    sample = FakeSample({"seg": seg})
    dset = _load(sample, {
        "embedding": "seg | ?",
        "affinity": "seg_out0 | seg | ?",      # different spec string, same resolved key
        "long_range": "seg | ?",
    })

    assert dset["embedding"] is dset["affinity"] is dset["long_range"]
    assert dset["embedding_mask"] is dset["affinity_mask"] is dset["long_range_mask"]
    assert sample.reads == ["seg"], f"seg read {len(sample.reads)}x: {sample.reads}"

    assert dset["embedding"].shape == PADDED
    expected = np.pad(seg, tuple((p, p) for p in PADDING), "constant")
    assert np.array_equal(dset["embedding"], expected)
    assert dset["embedding"].dtype == np.uint32
    # mask: no per-key mask and no shared mask -> all ones, then zero-padded
    expected_mask = np.pad(np.ones(SHAPE, "uint8"), tuple((p, p) for p in PADDING), "constant")
    assert np.array_equal(dset["embedding_mask"], expected_mask)


def test_different_transform_is_not_shared():
    seg = _seg()
    sample = FakeSample({"seg": seg})
    dset = _load(
        sample,
        {"embedding": "seg | ?", "mitochondria": "seg | ?"},
        requires_binarize=["mitochondria"],
    )

    assert dset["embedding"] is not dset["mitochondria"]
    assert dset["embedding"].dtype == np.uint32
    assert dset["mitochondria"].dtype == np.uint8
    assert np.array_equal(dset["mitochondria"], np.pad(
        (seg > 0).astype("uint8"), tuple((p, p) for p in PADDING), "constant"))
    # ...but the mask does not depend on the transform, so it IS shared
    assert dset["embedding_mask"] is dset["mitochondria_mask"]
    assert sample.reads == ["seg"], f"seg read {len(sample.reads)}x: {sample.reads}"


def test_semantic_mapping_ids_are_not_shared():
    seg = np.array([1, 2, 3] * (int(np.prod(SHAPE)) // 3), dtype="uint32").reshape(SHAPE)
    sample = FakeSample({"seg": seg})
    dset = _load(
        sample,
        {"dendrite": "seg | ?", "axon": "seg | ?", "soma": "seg | ?"},
        semantic_mapping={"dendrite": 1, "axon": 2, "soma": 3},
    )

    assert dset["dendrite"] is not dset["axon"] is not dset["soma"]
    pw = tuple((p, p) for p in PADDING)
    for name, sem_id in (("dendrite", 1), ("axon", 2), ("soma", 3)):
        assert np.array_equal(
            dset[name], np.pad((seg == sem_id).astype("uint8"), pw, "constant"))
    assert sample.reads == ["seg"], f"seg read {len(sample.reads)}x: {sample.reads}"


def test_combination_still_ors_its_parts():
    mye = np.zeros(SHAPE, dtype="uint32"); mye[0] = 1
    ecs = np.zeros(SHAPE, dtype="uint32"); ecs[1] = 1
    sample = FakeSample({"mye": mye, "ecs": ecs})
    dset = _load(sample, {"myelin": "mye + ecs | ?", "other": "ecs + mye | ?"})

    pw = tuple((p, p) for p in PADDING)
    assert np.array_equal(dset["myelin"], np.pad(np.maximum(mye, ecs), pw, "constant"))
    # order differs, so the normalized tuple differs -- these are NOT shared,
    # which is correct-but-conservative; the values still match.
    assert np.array_equal(dset["other"], dset["myelin"])


def test_zero_fill_negative_vs_unknown():
    sample = FakeSample({"seg": _seg()})
    dset = _load(
        sample,
        {"myelin": "mye | ?", "glia": "glia | ?", "fold": "fld | ?"},
        known_absent=["mye"],
    )

    # all three zero-fill, so the DATA array is one object
    assert dset["myelin"] is dset["glia"] is dset["fold"]
    assert dset["myelin"].shape == PADDED
    assert not dset["myelin"].any()

    # known_absent -> all-ones (real negative); unknown -> all-zeros (no loss)
    assert dset["myelin_mask"] is not dset["glia_mask"]
    assert dset["glia_mask"] is dset["fold_mask"]
    pw = tuple((p, p) for p in PADDING)
    assert np.array_equal(dset["myelin_mask"], np.pad(np.ones(SHAPE, "uint8"), pw, "constant"))
    assert not dset["glia_mask"].any()


def test_per_key_mask_is_read_and_shared():
    seg = _seg()
    mask = np.ones(SHAPE, dtype="uint8"); mask[0] = 0
    sample = FakeSample({"seg": seg}, masks={"seg": mask})
    dset = _load(sample, {"embedding": "seg | ?", "affinity": "seg | ?"})

    assert dset["embedding_mask"] is dset["affinity_mask"]
    pw = tuple((p, p) for p in PADDING)
    assert np.array_equal(dset["embedding_mask"], np.pad(mask, pw, "constant"))


def test_no_padding_still_shares():
    seg = _seg()
    sample = FakeSample({"seg": seg})
    image = np.zeros(SHAPE, dtype="uint8")
    with mock.patch.object(multi_zettaset, "CloudVolume") as cv:
        cv.return_value.__getitem__.return_value = _cv_order(image)
        dset = multi_zettaset.load_sample(
            sample, padding=(0, 0, 0), resolution=(8, 8, 8),
            zettaset_lookup={"embedding": "seg | ?", "affinity": "seg | ?"},
        )
    assert dset["embedding"] is dset["affinity"]
    assert np.array_equal(dset["embedding"], seg)


def _build_and_count(dset, names):
    """Run the sampler's build_dataset, recording which masks got loc=True.

    `loc=True` is what makes DataProvider3 describe a mask's sampleable region
    (bounding box, or an index array when sparse). Counting those calls tests
    build_dataset's dedup directly, independent of which representation
    DataProvider3 happens to choose.
    """
    from unittest.mock import patch as mpatch
    from dataprovider3 import Dataset as DPDataset
    from deepem.data.sampler.zettaset import Sampler

    spec = {"input": (1,) + PADDED}
    for name in names:
        spec[name] = (1,) + PADDED
        spec[name + "_mask"] = (1,) + PADDED

    loc_calls = []
    real_add_mask = DPDataset.add_mask

    def recording(self, key, data, offset=(0, 0, 0), loc=False):
        if loc:
            loc_calls.append(key)
        return real_add_mask(self, key, data, offset=offset, loc=loc)

    with mpatch.object(DPDataset, "add_mask", recording):
        ds = Sampler.build_dataset(None, "t", dset, spec)
    return ds, loc_calls, np.flatnonzero


def test_locs_computed_once_when_masks_are_equivalent():
    """4 mask keys, 2 distinct objects, but identical content -> one pass."""
    seg = _seg()
    sample = FakeSample({"seg": seg, "mit": (seg % 3 == 0).astype("uint32")})
    names = ("embedding", "affinity", "long_range", "mitochondria")
    dset = _load(
        sample,
        {"embedding": "seg | ?", "affinity": "seg | ?",
         "long_range": "seg | ?", "mitochondria": "mit | ?"},
        requires_binarize=["mitochondria"],
    )
    # precondition: 2 distinct objects, equal content (both all-ones, padded)
    assert dset["embedding_mask"] is not dset["mitochondria_mask"]
    assert np.array_equal(dset["embedding_mask"], dset["mitochondria_mask"])

    ds, loc_calls, real = _build_and_count(dset, names)
    assert len(loc_calls) == 1, f"loc=True for {loc_calls}, expected 1 mask"
    # one distinct mask -> described by its bounding box, no index array
    assert ds.locs["data"] is None
    assert ds.locs["count"] == int(np.count_nonzero(dset["embedding_mask"]))
    for name in names:
        assert name + "_mask" in ds.data


def test_locs_still_unions_genuinely_different_masks():
    """A mask with different content must still contribute its locations."""
    seg = _seg()
    seg_mask = np.ones(SHAPE, dtype="uint8"); seg_mask[0] = 0     # drops z=0
    mit_mask = np.ones(SHAPE, dtype="uint8"); mit_mask[-1] = 0    # drops z=-1
    sample = FakeSample(
        {"seg": seg, "mit": (seg % 3 == 0).astype("uint32")},
        masks={"seg": seg_mask, "mit": mit_mask},
    )
    names = ("embedding", "affinity", "mitochondria")
    dset = _load(
        sample,
        {"embedding": "seg | ?", "affinity": "seg | ?", "mitochondria": "mit | ?"},
        requires_binarize=["mitochondria"],
    )
    assert not np.array_equal(dset["embedding_mask"], dset["mitochondria_mask"])

    ds, loc_calls, real = _build_and_count(dset, names)
    assert len(loc_calls) == 2, f"loc=True for {loc_calls}, expected 2 masks"
    # a second distinct mask forces the exact index-array union
    expected = np.union1d(real(dset["embedding_mask"]), real(dset["mitochondria_mask"]))
    assert np.array_equal(ds.locs["data"], expected)
    assert ds.locs["count"] == expected.size


def test_locs_equal_count_but_different_content_is_not_merged():
    """count_nonzero is only a discriminator -- array_equal must decide."""
    seg = _seg()
    seg_mask = np.ones(SHAPE, dtype="uint8"); seg_mask[0] = 0
    mit_mask = np.ones(SHAPE, dtype="uint8"); mit_mask[1] = 0     # same nnz, different set
    sample = FakeSample(
        {"seg": seg, "mit": seg.copy()},
        masks={"seg": seg_mask, "mit": mit_mask},
    )
    dset = _load(sample, {"embedding": "seg | ?", "mitochondria": "mit | ?"},
                 requires_binarize=["mitochondria"])
    a, b = dset["embedding_mask"], dset["mitochondria_mask"]
    assert np.count_nonzero(a) == np.count_nonzero(b) and not np.array_equal(a, b)

    ds, loc_calls, real = _build_and_count(dset, ("embedding", "mitochondria"))
    assert len(loc_calls) == 2, f"loc=True for {loc_calls}, expected 2 masks"
    assert np.array_equal(ds.locs["data"], np.union1d(real(a), real(b)))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
