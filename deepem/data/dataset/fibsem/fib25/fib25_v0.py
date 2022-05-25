import numpy as np
import os

import dataprovider3.emio as emio


fib25_dir = 'FIB-25'
data_keys = ['validation_sample']
merger_ids = [2148]


def load_data(base_dir , data_ids=None, **kwargs):
    if data_ids is None:
        return {}
    
    base_dir = os.path.expanduser(base_dir)
    data_dir = os.path.join(base_dir, fib25_dir)

    data = {}
    for data_id in data_ids:
        if data_id in data_keys:
            dpath = os.path.join(data_dir, data_id)
            assert os.path.exists(dpath)
            data[data_id] = load_dataset(dpath, **kwargs)
    return data


def load_dataset(dpath, **kwargs):
    dset = {}

    # Image
    fpath = os.path.join(dpath, "img.h5")
    print(fpath)
    dset['img'] = emio.imread(fpath).astype(np.float32)
    dset['img'] /= 255.0

    # Segmentation
    fpath = os.path.join(dpath, "seg.h5")
    print(fpath)
    dset['seg'] = emio.imread(fpath).astype(np.uint8)

    # Additoinal info
    dset['loc'] = True

    # Mask out mergers
    dset['msk'] = np.ones(dset['seg'].shape, dtype=np.uint8)
    if len(merger_ids) > 0:
        idx = np.isin(dset['seg'], merger_ids)
        dset['msk'][idx] = 0

    return dset