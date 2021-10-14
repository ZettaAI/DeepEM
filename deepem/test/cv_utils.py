import numpy as np

import cloudvolume as cv
from cloudvolume.lib import Vec, Bbox
from taskqueue import LocalTaskQueue
import igneous
from igneous.task_creation import create_downsampling_tasks

from deepem.utils import py_utils


def make_info(num_channels, layer_type, dtype, shape, resolution,
              offset=(0,0,0), chunk_size=(64,64,64)):
    return cv.CloudVolume.create_new_info(
        num_channels, layer_type, dtype, 'raw', resolution, offset, shape,
        chunk_size=chunk_size)

def get_coord_bbox(cvol, opt):
    # Cutout
    offset = cvol.voxel_offset
    if opt.center is not None:
        assert opt.size is not None
        opt.begin = tuple(x - (y//2) for x, y in zip(opt.center, opt.size))
        opt.end = tuple(x + y for x, y in zip(opt.begin, opt.size))
    else:
        if not opt.begin:
            opt.begin = offset
        if not opt.end:
            if not opt.size:
                opt.end = offset + cvol.shape
            else:
                opt.end = tuple(x + y for x, y in zip(opt.begin, opt.size))
    return Bbox(opt.begin, opt.end)

def cutout(opt, gs_path, dtype='uint8'):
    if '{}' in gs_path:
        gs_path = gs_path.format(*opt.keywords)
    print(gs_path)

    # CloudVolume.
    cvol = cv.CloudVolume(gs_path, mip=opt.in_mip, cache=opt.cache,
                          fill_missing=True, parallel=opt.parallel)

    # Based on MIP level args are specified in
    coord_bbox = get_coord_bbox(cvol, opt)
    if opt.coord_mip != opt.in_mip:
        print(f"mip {opt.coord_mip} = {coord_bbox}")
    # Based on in_mip
    in_bbox = cvol.bbox_to_mip(coord_bbox, mip=opt.coord_mip, to_mip=opt.in_mip)
    print(f"mip {opt.in_mip} = {in_bbox}")
    cutout = cvol[in_bbox.to_slices()]

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

    # Use CloudVolume to make sure the output bbox matches the input.
    # MIP hierarchies are not guaranteed to be powers of 2 (especially
    # with float resolutions), so need to use the info file and be 
    # consistent computing the offset between input and output.
    gs_path = opt.gs_input
    if '{}' in gs_path:
        gs_path = gs_path.format(*opt.keywords)
    in_vol = cv.CloudVolume(gs_path, mip=opt.in_mip, cache=opt.cache,
                          fill_missing=True, parallel=opt.parallel)
    coord_bbox = get_coord_bbox(in_vol, opt)
    # Offset is defined at coord_mip, so adjust coord_bbox first
    if opt.offset:
        start_adjust = coord_bbox.minpt - opt.offset
        coord_bbox -= start_adjust
    in_bbox = in_vol.bbox_to_mip(coord_bbox, mip=opt.coord_mip, to_mip=opt.in_mip)

    # Patch offset correction (when output patch is smaller than input patch)
    patch_offset = (np.array(opt.inputsz) - np.array(opt.outputsz)) // 2
    patch_offset = Vec(*np.flip(patch_offset, 0))
    in_bbox.minpt += patch_offset
    # in_bbox.stop -= 2*patch_offset # using the data to define shape

    # Create info
    info = make_info(num_channels, 'image', str(data.dtype), shape,
                     opt.resolution, offset=in_bbox.minpt, chunk_size=opt.chunk_size)
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
        with LocalTaskQueue(parallel=opt.parallel) as tq:
            tasks = create_downsampling_tasks(gs_path, mip=0, fill_missing=True)
            tq.insert_all(tasks)
