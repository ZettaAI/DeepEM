import numpy as np
import os

import dataprovider3.emio as emio


data_keys = [f"cutout00{i+1}" for i in range(6)]


def load_data(data_dir, data_ids=None, **kwargs):
    if data_ids is None:
        return {}

    data_dir = os.path.expanduser(data_dir)

    data = {}
    for data_id in data_ids:
        if data_id in data_keys:
            dpath = os.path.join(data_dir, data_id)
            assert os.path.exists(dpath)
            data[data_id] = load_dataset(dpath, **kwargs)

    return data


def load_dataset(dpath, class_keys=[], **kwargs):
    dset = {}

    # Image
    fpath = os.path.join(dpath, "img.h5")
    print(fpath)
    dset['img'] = emio.imread(fpath).astype('float32')
    dset['img'] /= 255.0

    # Mask
    fpath = os.path.join(dpath, "msk.h5")
    print(fpath)
    dset['msk'] = emio.imread(fpath).astype('uint8')

    # Segmentation
    if ('aff' in class_keys) or ('long' in class_keys):
        fpath = os.path.join(dpath, "seg.h5")
        print(fpath)
        dset['seg'] = emio.imread(fpath).astype('uint16')

    # Myelin (optional)
    if 'mye' in class_keys:
        fpath = os.path.join(dpath, "mye.h5")
        print(fpath)
        dset['mye'] = emio.imread(fpath).astype('uint8')

    # Additoinal info
    dset['loc'] = True

    return dset