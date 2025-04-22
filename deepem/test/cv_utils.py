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
    offset = cvol.voxel_offset
    volume_shape = cvol.shape[:3]

    if opt.center is not None:
        assert opt.size is not None
        opt.begin = tuple(x - (y // 2) for x, y in zip(opt.center, opt.size))
        opt.end = tuple(x + y for x, y in zip(opt.begin, opt.size))
    else:
        if opt.begin is None:
            opt.begin = offset  # Default to voxel_offset if not provided

        if opt.end is None:
            if opt.size is None:
                # When size is not specified, set end to the end of the dataset
                opt.end = tuple(o + s for o, s in zip(offset, volume_shape))
            else:
                # When size is specified, calculate end based on begin + size
                opt.end = tuple(b + s for b, s in zip(opt.begin, opt.size))

    return Bbox(opt.begin, opt.end)

def cutout(opt, gs_path, dtype='uint8', channels=0, in_mip=None, coord_mip=None):
    if '{}' in gs_path:
        gs_path = gs_path.format(*opt.keywords)
    print(gs_path)

    # Use provided mip values if available, otherwise fall back to opt values
    actual_coord_mip = coord_mip if coord_mip is not None else opt.coord_mip
    actual_in_mip = in_mip if in_mip is not None else opt.in_mip

    # CloudVolume for coordinate handling (coord_mip)
    coord_cvol = cv.CloudVolume(gs_path, mip=actual_coord_mip)

    # CloudVolume for data fetching (in_mip)
    data_cvol = cv.CloudVolume(gs_path, mip=actual_in_mip, cache=opt.cache,
                              fill_missing=True, parallel=opt.parallel)

    # Get bounding box based on coord_mip
    coord_bbox = get_coord_bbox(coord_cvol, opt)

    if actual_coord_mip != actual_in_mip:
        print(f"mip {actual_coord_mip} = {coord_bbox}")

    # Convert bbox to in_mip coordinates if needed
    in_bbox = coord_cvol.bbox_to_mip(coord_bbox, mip=actual_coord_mip, to_mip=actual_in_mip)
    print(f"mip {actual_in_mip} = {in_bbox}")

    # Data cutout from in_mip
    cutout = data_cvol[in_bbox.to_slices()]

    # Transpose & squeeze
    cutout = cutout.transpose([3, 2, 1, 0])
    
    # Slice channels if specified
    if channels > 0:
        cutout = cutout[:channels, ...]

    return np.squeeze(cutout).astype(dtype)


def ingest(data, opt, tag=None):
    # Neuroglancer format
    data = py_utils.to_tensor(data)
    data = data.transpose((3, 2, 1, 0))
    num_channels = data.shape[-1]
    shape = data.shape[:-1]

    # Use CloudVolume with coord_mip for coordinate handling
    gs_path = opt.gs_input
    if '{}' in gs_path:
        gs_path = gs_path.format(*opt.keywords)

    coord_cvol = cv.CloudVolume(gs_path, mip=opt.coord_mip)

    # Get bounding box in coord_mip
    coord_bbox = get_coord_bbox(coord_cvol, opt)

    # Offset is defined at coord_mip, so adjust coord_bbox first
    if opt.offset:
        start_adjust = coord_bbox.minpt - opt.offset
        coord_bbox -= start_adjust

    # Convert bbox to in_mip coordinates
    in_bbox = coord_cvol.bbox_to_mip(coord_bbox, mip=opt.coord_mip, to_mip=opt.in_mip)

    # Patch offset correction (when output patch is smaller than input patch)
    patch_offset = (0, 0, 0)
    if opt.tilt_series > 0:
        if opt.tilt_series_crop is not None:
            outputsz = np.array(opt.fov) * np.array(opt.scale)
            patch_offset = (outputsz - np.array(opt.tilt_series_crop)) // 2
    else:
        patch_offset = (np.array(opt.inputsz) - np.array(opt.outputsz)) // 2

    patch_offset = Vec(*np.flip(patch_offset, 0))

     # Create info using the adjusted offset
    offset = in_bbox.minpt + patch_offset
    info = make_info(num_channels, 'image', str(data.dtype), shape,
                     opt.resolution, offset=offset, chunk_size=opt.chunk_size)
    print(info)

    # Output path formatting
    gs_path = opt.gs_output
    if '{}' in opt.gs_output:
        if opt.keywords:
            gs_path = gs_path.format(*opt.keywords)
        else:
            if opt.center is not None:
                coord = "x{}_y{}_z{}".format(*opt.center)
                coord += "_s{}-{}-{}".format(*opt.size)
            else:
                coord = '_'.join([f"{b}-{e}" for b, e in zip(opt.begin, opt.end)])
            gs_path = gs_path.format(coord)

    # Tagging
    if tag is not None:
        gs_path = gs_path.rstrip('/') + '/' + tag

    print(f"gs_output:\n{gs_path}")
    data_vol = cv.CloudVolume(gs_path, mip=0, info=info, parallel=opt.parallel)
    data_vol[:, :, :, :] = data
    data_vol.commit_info()

    # Downsample
    if opt.downsample:
        with LocalTaskQueue(parallel=opt.parallel) as tq:
            tasks = create_downsampling_tasks(
                gs_path,
                mip=0,
                fill_missing=True,
                factor=opt.downsample_factor,
            )
            tq.insert_all(tasks)
