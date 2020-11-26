import numpy as np

import cloudvolume as cv
from cloudvolume.lib import Vec, Bbox
from taskqueue import LocalTaskQueue

from deepem.utils import py_utils


def make_info(num_channels, layer_type, dtype, shape, resolution,
              offset=(0,0,0), chunk_size=(64,64,64)):
    return cv.CloudVolume.create_new_info(
        num_channels, layer_type, dtype, 'raw', resolution, offset, shape,
        chunk_size=chunk_size)


def cutout(opt, gs_path, dtype='uint8'):
    if '{}' in gs_path:
        gs_path = gs_path.format(*opt.keywords)
    print(gs_path)

    # CloudVolume.
    cvol = cv.CloudVolume(gs_path, mip=opt.in_mip, cache=opt.cache,
                          fill_missing=True, parallel=opt.parallel)

    # Cutout
    offset0 = cvol.mip_voxel_offset(0)
    if opt.center is not None:
        assert opt.size is not None
        opt.begin = tuple(x - (y//2) for x, y in zip(opt.center, opt.size))
        opt.end = tuple(x + y for x, y in zip(opt.begin, opt.size))
    else:
        if not opt.begin:
            opt.begin = offset0
        if not opt.end:
            if not opt.size:
                opt.end = offset0 + cvol.mip_volume_size(0)
            else:
                opt.end = tuple(x + y for x, y in zip(opt.begin, opt.size))
    sl = [slice(x,y) for x, y in zip(opt.begin, opt.end)]
    print(f"begin = {opt.begin}")
    print(f"end = {opt.end}")

    # Coordinates
    print(f"mip 0 = {sl}")
    sl = cvol.slices_from_global_coords(sl)
    print(f"mip {opt.in_mip} = {sl}")
    cutout = cvol[sl]

    # Transpose & squeeze
    cutout = cutout.transpose([3,2,1,0])
    cutout = np.squeeze(cutout).astype(dtype)
    return cutout


def ingest(data, opt, tag=None):
    # Neuroglancer format
    data = py_utils.to_tensor(data)
    data = data.transpose((3,2,1,0))
    num_channels = data.shape[-1]
    shape = data.shape[:-1]

    # Offset
    if opt.offset is None:
        opt.offset = opt.begin

    # MIP level correction
    if opt.gs_input and opt.in_mip > 0:
        o = opt.offset
        p = pow(2,opt.in_mip)
        offset = (o[0]//p, o[1]//p, o[2])
    else:
        offset = opt.offset

    # Patch offset correction (when output patch is smaller than input patch)
    patch_offset = (np.array(opt.inputsz) - np.array(opt.outputsz)) // 2
    offset = tuple(np.array(offset) + np.flip(patch_offset, 0))

    # Create info
    info = make_info(num_channels, 'image', str(data.dtype), shape,
                     opt.resolution, offset=offset, chunk_size=opt.chunk_size)
    print(info)
    gs_path = opt.gs_output
    if '{}' in opt.gs_output:
        if opt.keywords:
            gs_path = gs_path.format(*opt.keywords)
        else:
            if opt.center is not None:
                coord = "x{}_y{}_z{}".format(*opt.center)
                coord += "_s{}-{}-{}".format(*opt.size)
            else:
                coord = '_'.join([f"{b}-{e}" for b,e in zip(opt.begin,opt.end)])
            gs_path = gs_path.format(coord)

    # Tagging
    if tag is not None:
        if gs_path[-1] == '/':
            gs_path += tag
        else:
            gs_path += ('/' + tag)

    print(f"gs_output:\n{gs_path}")
    cvol = cv.CloudVolume(gs_path, mip=0, info=info,
                          parallel=opt.parallel)
    cvol[:,:,:,:] = data
    cvol.commit_info()

    # Downsample
    if opt.downsample:
        import igneous
        from igneous.task_creation import create_downsampling_tasks

        with LocalTaskQueue(parallel=opt.parallel) as tq:
            # create_downsampling_tasks(tq, gs_path, mip=0, fill_missing=True)
            tasks = create_downsampling_tasks(gs_path, mip=0, fill_missing=True)
            tq.insert_all(tasks)
