import os

import numpy as np
import cloudvolume as cv
from cloudfiles import CloudFiles


hemibrain_dir = 'gs://zetta-prieto-godino-fly-larva-001-seg-temp/seg-dataset/hemibrain'
larva_dir = 'gs://zetta-prieto-godino-fly-larva-001-seg-temp/seg-dataset/larva/lensoftruth-more-context'
focused_dir = 'gs://zetta-prieto-godino-fly-larva-001-seg-temp/seg-dataset/larva/lensoftruth-focused'
data_keys = (
    [f"h{i:03d}" for i in range(8)]  # hemibrain
    + [f"l{i:03d}" for i in range(13)]  # larva
    + [f"f{i:03d}" for i in range(3)]  # larva focused annotation
)

def load_data(base_dir, data_ids=None, **kwargs):
    if data_ids is None:
        return {}

    data = {}
    for data_id in data_ids:
        if data_id in data_keys:

            if data_id.startswith("h"):  # hemibrain
                samplepath = os.path.join(hemibrain_dir, data_id[1:])
            elif data_id.startswith("l"):  # larva
                samplepath = os.path.join(larva_dir, data_id[1:])
            elif data_id.startswith("f"):  # larva focused annotation
                samplepath = os.path.join(focused_dir, data_id[1:])

            cf = CloudFiles(samplepath)
            sampleinfo = cf.get_json("info")

            data[data_id] = load_dataset(samplepath, sampleinfo, **kwargs)

        else:
            raise KeyError(f"invalid data id: {data_id}")

    return data


def load_dataset(dpath, info, **kwargs):
    dset = {}

    # Image
    vers = "000"
    fpath = os.path.join(dpath, "image", vers)
    print(fpath)
    cloudvol = cv.CloudVolume(fpath, cache=True, mip=(16, 16, 16))
    dset['img'] = cloudvol[:].transpose(3, 2, 1, 0)[0, ...]
    dset['img'] = (dset['img'] / 255.).astype(np.float32)

    # Segmentation
    vers = sorted(info["annotations"]["seg"]["versions"].keys())[-1]
    fpath = os.path.join(dpath, "seg", vers)
    print(fpath)
    cloudvol = cv.CloudVolume(fpath, cache=True, mip=(16, 16, 16))
    cloudvol.fill_missing = True
    dset['seg'] = cloudvol[:].transpose(3, 2, 1, 0)[0, ...]

    # Additional info
    dset['loc'] = True

    # Mask
    seg = dset['seg']
    dset['msk'] = np.zeros(seg.shape, dtype=np.uint8)
    if "hemibrain" in dpath:
        dset['msk'][64:-64, 64:-64, 64:-64] = 1

    elif "focused" in dpath:
        # manually drawn mask
        fpath = os.path.join(dpath, "seg", f"{vers}_mask")
        cloudvol = cv.CloudVolume(fpath, cache=True, mip=(16, 16, 16))
        cloudvol.fill_missing = True
        dset["msk"] = cloudvol[:].transpose(3, 2, 1, 0)[0, ...]

    else:  # larva dataset
        if dpath.endswith("002"):
            dset['msk'][190:-190, 185:-195, 190:-190] = 1
        elif dpath[-3:] in ["006", "010", "012"]:
            dset['msk'][171:-171, 171:-171, 171:-171] = 1
        elif dpath.endswith("007"):
            dset['msk'][160:-160, 160:-160, 160:-160] = 1
        elif dpath[-3:] in ["004", "005", "008"]:
            dset['msk'][172:-172, 172:-172, 172:-172] = 1
        else:
            dset['msk'][190:-190, 190:-190, 190:-190] = 1

    # unknown/unclear segment
    dset["msk"][seg == 999] = 0

    return dset


if __name__ == "__main__":
    data = load_data(None, data_ids=["000"])

    print(data["000"]["img"].shape)
    print(data["000"]["seg"].shape)
    print(data["000"]["msk"].shape)
