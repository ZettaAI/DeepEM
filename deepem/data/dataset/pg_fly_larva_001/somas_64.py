import os

import numpy as np
import cloudvolume as cv
from cloudfiles import CloudFiles


img_cvpath = "gs://zetta-prieto-godino-fly-larva-001-image/image-v1-iso"
dataset_dir = "gs://zetta-prieto-godino-fly-larva-001-seg-temp/seg-dataset/somata"
data_keys = (
    "6388f1170100009d0023f9ef",
    "6388eed00100009d0023f9e6",
    "6388f0bf010000930023f9ed",
    "6388ef8f010000a80023f9ea",
    "6388f198010000200023f9f1",
    "6388ef1a0100009d0023f9e8",
    "639246b8010000a20023faf1",
    "6388ee4d010000a60023f9e4",
    "63924770010000200023faf3",
    "6392483b010000a70023faf6",
    "639248a0010000930023faf8",
    "63924992010000a60023fafa",
)


def load_data(base_dir, data_ids=None, **kwargs):
    if data_ids is None:
        return {}

    data = {}
    for data_id in data_ids:
        if data_id in data_keys:

            samplepath = os.path.join(dataset_dir, data_id)

            cf = CloudFiles(samplepath)
            sampleinfo = cf.get_json("info")

            data[data_id] = load_dataset(samplepath, sampleinfo, **kwargs)

        else:
            raise KeyError(f"invalid data id: {data_id}")

    return data


def load_dataset(dpath, info, **kwargs):
    dset = {}

    # Segmentation
    vers = sorted(info["annotations"]["seg"]["versions"].keys())[-1]
    fpath = os.path.join(dpath, "seg", vers)
    print(fpath)
    cloudvol = cv.CloudVolume(fpath, cache=True, mip=(64, 64, 64))
    cloudvol.fill_missing = True
    dset['seg'] = cloudvol[:].transpose(3, 2, 1, 0)[0, ...]
    bbox = cloudvol.bounds

    # Image
    vers = "000"
    print(img_cvpath)
    cloudvol = cv.CloudVolume(img_cvpath, cache=True, mip=(64, 64, 64))
    dset['img'] = cloudvol[bbox].transpose(3, 2, 1, 0)[0, ...]
    dset['img'] = (dset['img'] / 255.).astype(np.float32)


    # Additional info
    dset['loc'] = True

    # Mask
    fpath = os.path.join(dpath, "seg", f"{vers}_mask")
    print(fpath)
    cloudvol = cv.CloudVolume(fpath, cache=True, mip=(64, 64, 64))
    cloudvol.fill_missing = True
    dset['msk'] = cloudvol[:].transpose(3, 2, 1, 0)[0, ...]

    # unknown/unclear segment
    dset["msk"][dset["seg"] == 999] = 0

    return dset


if __name__ == "__main__":
    data = load_data(None, data_ids=["000"])

    print(data["000"]["img"].shape)
    print(data["000"]["seg"].shape)
    print(data["000"]["msk"].shape)
