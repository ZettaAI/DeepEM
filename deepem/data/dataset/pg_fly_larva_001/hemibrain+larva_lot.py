import os

import numpy as np
import cloudvolume as cv
from cloudfiles import CloudFiles


hemibrain_dir = 'gs://zetta-prieto-godino-fly-larva-001-seg-temp/seg-dataset/hemibrain'
larva_dir = 'gs://zetta-prieto-godino-fly-larva-001-seg-temp/seg-dataset/larva/lensoftruth'
data_keys = [f"h{i:03d}" for i in range(8)] + [f"l{i:03d}" for i in range(4)]


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
    dset['img'] = cv.CloudVolume(fpath, cache=True)[:].transpose(3, 2, 1, 0)[0, ...]
    dset['img'] = (dset['img'] / 255.).astype(np.float32)

    # Segmentation
    vers = sorted(info["annotations"]["seg"]["versions"].keys())[-1]
    fpath = os.path.join(dpath, "seg", vers)
    print(fpath)
    dset['seg'] = cv.CloudVolume(fpath, cache=True)[:].transpose(3, 2, 1, 0)[0, ...]

    # Additoinal info
    dset['loc'] = True

    # Mask
    seg = dset['seg']
    dset['msk'] = np.zeros(seg.shape, dtype=np.uint8)
    # hack for one mismatched bbox
    if dpath.endswith("larva/lensoftruth/002"):
        dset['msk'][128:-128, 138:-138, 128:-128] = 1
    else:
        dset['msk'][128:-128, 128:-128, 128:-128] = 1

    return dset


if __name__ == "__main__":
    data = load_data(None, data_ids=["000"])

    print(data["000"]["img"].shape)
    print(data["000"]["seg"].shape)
    print(data["000"]["msk"].shape)
