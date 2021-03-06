import numpy as np
import os

import dataprovider3.emio as emio


# Basil dataset
data_dir = 'basil/ground_truth'
basil_dir = 'mip1/padded_x512_y512_z32'
basil_keys = [f"basil{i+1:03d}" for i in range(11)]
basil_info = {
    'img': 'img.h5',
    'seg': 'seg.h5',
    'msk': 'msk.h5',
    'mye': 'mye.h5',
    'blv': 'blv.h5',
    'loc': True,
}


def load_data(base_dir, data_ids=None, **kwargs):
    if data_ids is None:
        return {}
    
    base_dir = os.path.expanduser(base_dir)
    base_dir = os.path.join(base_dir, data_dir)

    data = {}
    for data_id in data_ids:
        if data_id == 'basil_superset':
            # Superset
            superset = {}
            for dkey in basil_keys:
                dpath = os.path.join(base_dir, basil_dir, dkey)
                if os.path.exists(dpath):
                    superset[dkey] = load_dataset(dpath, tag=dkey, **kwargs)
            data['basil_superset'] = superset
        else:
            # Individual dataset
            if data_id in basil_keys:
                dpath = os.path.join(base_dir, basil_dir, data_id)
                assert os.path.exists(dpath)
                data[data_id] = load_dataset(dpath, tag=data_id, **kwargs)
    return data


def load_dataset(dpath, tag, class_keys=[], **kwargs):
    assert len(class_keys) > 0
    dset = {}
    
    # Image
    fpath = os.path.join(dpath, basil_info['img'])
    print(fpath)
    dset['img'] = emio.imread(fpath).astype(np.float32)
    dset['img'] /= 255.0

    # Mask
    if tag in ['basil001','basil002']:
        fpath = os.path.join(dpath, 'msk.d128.h5')
    else:
        fpath = os.path.join(dpath, basil_info['msk'])
    print(fpath)
    dset['msk'] = emio.imread(fpath).astype(np.uint8)

    # Segmentation
    if 'aff' in class_keys or 'long' in class_keys:
        fpath = os.path.join(dpath, basil_info['seg'])
        print(fpath)
        dset['seg'] = emio.imread(fpath).astype(np.uint32)

    # Myelin (optional)
    if 'mye' in class_keys:
        fpath = os.path.join(dpath, basil_info['mye'])
        if os.path.exists(fpath):
            print(fpath)
            dset['mye'] = emio.imread(fpath).astype(np.uint8)
        else:
            assert 'msk' in dset
            dset['mye'] = np.zeros_like(dset['msk'])

    # Blood vessel (optional)
    if 'blv' in class_keys:
        fpath = os.path.join(dpath, basil_info['blv'])
        if os.path.exists(fpath):
            print(fpath)
            dset['blv'] = emio.imread(fpath).astype(np.uint8)
        else:
            assert 'msk' in dset
            dset['blv'] = np.zeros_like(dset['msk'])

    # Additoinal info
    dset['loc'] = basil_info['loc']

    return dset
