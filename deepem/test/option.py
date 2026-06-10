import argparse
from collections import OrderedDict
import json
import math
import os
import numpy as np

from deepem.utils.py_utils import vec3, vec3f


class Options(object):
    """
    Test options.
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--exp_name', required=True)
        self.parser.add_argument('--chkpt_num', type=int, default=0)
        self.parser.add_argument('--gpu_id', type=str, default='0')

        # CPU inference
        self.parser.add_argument('--cpu', action='store_true')

        # cuDNN auto-tuning
        self.parser.add_argument('--no_autotune', action='store_false')

        # Overwriting
        self.parser.add_argument('--model', default=None)
        self.parser.add_argument('--pretrain', action='store_true')
        self.parser.add_argument('--no_eval', action='store_true')
        self.parser.add_argument('--inputsz', type=vec3, default=None)
        self.parser.add_argument('--outputsz', type=vec3, default=None)
        self.parser.add_argument('--force_crop', type=vec3, default=None)
        self.parser.add_argument('--fov', type=vec3, default=None)
        self.parser.add_argument('--depth', type=int, default=4)
        self.parser.add_argument('--width', type=int, default=None, nargs='+')
        self.parser.add_argument('--group', type=int, default=0)
        self.parser.add_argument('--group_eps', type=float, default=1e-5)
        self.parser.add_argument('--act', default='ReLU')
        self.parser.add_argument('--updown_scale_factor', type=vec3f, default=None)

        # Tilt-series electron tomography
        self.parser.add_argument('--tilt_series', type=int, default=0)
        self.parser.add_argument('--tilt_series_in', type=int, default=12)
        self.parser.add_argument('--tilt_series_out', type=int, default=4)
        self.parser.add_argument('--tilt_series_crop', type=vec3, default=None)

        # Metric learning
        self.parser.add_argument('--vec', type=int, default=0)
        self.parser.add_argument('--vec_to', default=None)  # 'aff' or 'pca'
        self.parser.add_argument('--edges', type=vec3, default=[(0,0,1),(0,1,0),(1,0,0)], nargs='+')
        self.parser.add_argument('--delta_d', type=float, default=1.5)
        self.parser.add_argument('--scale_init', type=float, default=1.0)

        # Per-class enable flags — auto-declared from the central registry.
        # `vec` and `long` take integer channel counts (legacy), so they are
        # declared explicitly and skipped here.
        from deepem.data.classes import REGISTRY as CLASS_REGISTRY
        for flag_name in CLASS_REGISTRY:
            if flag_name in ('vec', 'long'):
                continue
            self.parser.add_argument(f'--{flag_name}', action='store_true')

        self.parser.add_argument('--long', type=int, default=0)
        self.parser.add_argument('--aff_deprecated', type=int, default=None)
        self.parser.add_argument('--mye_thresh', type=float, default=0.5)
        self.parser.add_argument('--blv_num_channels', type=int, default=1)
        self.parser.add_argument('--sem',  action='store_true')
        self.parser.add_argument('--merge_classes', type=str, default=[], nargs='+')

        # Semantic segmentation
        self.parser.add_argument('--semantic', action='store_true')

        # Test-time augmentation
        self.parser.add_argument('--test_aug', type=int, default=None, nargs='+')
        self.parser.add_argument('--test_aug16', action='store_true')
        self.parser.add_argument('--test_aug64', action='store_true')
        self.parser.add_argument('--variance', action='store_true')

        # Temperature T for softer softmax
        self.parser.add_argument('--temperature', type=float, default=None)

        # Cloud-volume input
        self.parser.add_argument('--gs_inputs', type=json.loads, default={})
        self.parser.add_argument('--gs_input', default='')
        self.parser.add_argument('--gs_input_mask', default='')
        self.parser.add_argument('--gs_input_norm', type=float, default=None, nargs='+')
        self.parser.add_argument('--gs_normalize_keys', type=str, nargs='+', default=['input'])
        self.parser.add_argument('--in_mip', type=int, default=0)
        self.parser.add_argument('--coord_mip', type=int, default=0)
        self.parser.add_argument('--in_mips', type=json.loads, default={})
        self.parser.add_argument('--coord_mips', type=json.loads, default={})
        self.parser.add_argument('--cache', action='store_true')
        self.parser.add_argument('-b','--begin', type=vec3, default=None)
        self.parser.add_argument('-e','--end', type=vec3, default=None)
        self.parser.add_argument('-c','--center', type=vec3, default=None)
        self.parser.add_argument('-s','--size', type=vec3, default=None)        

        # Cloud-volume output
        self.parser.add_argument('--gs_output', default='')
        self.parser.add_argument('--tags', type=json.loads, default=None)
        self.parser.add_argument('--keywords', default=[], nargs='+')
        self.parser.add_argument('-p','--parallel', type=int, default=16)
        self.parser.add_argument('-d','--downsample', action='store_true')
        self.parser.add_argument('--downsample_factor', type=vec3, default=(2,2,1))
        self.parser.add_argument('-r','--resolution', type=vec3, default=(4,4,40))
        self.parser.add_argument('-o','--offset', type=vec3, default=None)
        self.parser.add_argument('--chunk_size', type=vec3, default=(64,64,16))

        # Data
        self.parser.add_argument('--data_dir', default="")
        self.parser.add_argument('--data_names', nargs='+')
        self.parser.add_argument('--input_name', default="img.h5")

        # Forward scanning
        self.parser.add_argument('--out_prefix', default='')
        self.parser.add_argument('--out_tag', default='')
        self.parser.add_argument('--overlap', type=vec3f, default=(0.5,0.5,0.5))
        self.parser.add_argument('--stride', type=vec3, default=None)
        self.parser.add_argument('--scale', type=vec3, default=(1,1,1))
        self.parser.add_argument('--mirror', type=vec3, default=None)
        self.parser.add_argument('--crop_border', type=vec3, default=None)
        self.parser.add_argument('--crop_center', type=vec3, default=None)
        self.parser.add_argument('--blend', default='bump')
        self.parser.add_argument('--bump', default='zung')  # 'zung'/'wu'/'wu_no_crust'

        # Asymmetric mask
        self.parser.add_argument('--mask_edges', type=vec3, default=[(0,0,1),(0,1,0),(1,0,0)], nargs='+')

        # Benchmark
        self.parser.add_argument('--dummy', action='store_true')
        self.parser.add_argument('--dummy_inputsz', type=int, default=[128,1024,1024], nargs='+')

        # Mixed-precision inference
        self.parser.add_argument('--mixed_precision', type=str, default=None, choices=['fp16', 'bf16'])

        # Super-resolution
        self.parser.add_argument('--sr_mode', action='store_true')
        self.parser.add_argument('--sr_scale_z', type=int, default=5)
        self.parser.add_argument('--sr_input_iso', action='store_true')

        # Export to ONNX
        self.parser.add_argument('--onnx', action='store_true')
        self.parser.add_argument('--opset_version', type=int, default=10)

        # Channel voting (semantic segmentation post-proc)
        self.parser.add_argument('--channel_voting', type=str, default=None,
                                 choices=['semantic_map', 'argmax'])
        self.parser.add_argument('--channel_voting_tag', type=str, default='semantic')
        self.parser.add_argument('--keep_per_class', action='store_true')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()

        # Device
        opt.device = 'cpu' if opt.cpu else 'cuda'

        # Directories
        if opt.exp_name.split('/')[0] == 'experiments':
            opt.exp_dir = opt.exp_name
        else:
            opt.exp_dir = f"experiments/{opt.exp_name}"
        opt.model_dir = os.path.join(opt.exp_dir, 'models')
        opt.fwd_dir = os.path.join(opt.exp_dir, 'forward')

        # Model spec
        opt.fov = tuple(opt.fov)
        opt.inputsz = opt.fov if opt.inputsz is None else opt.inputsz
        opt.outputsz = opt.fov if opt.outputsz is None else opt.outputsz
        in_channels = opt.tilt_series if opt.tilt_series > 0 else 1
        opt.in_spec = dict(input=(in_channels,) + opt.inputsz)
        opt.out_spec = dict()

        # Tilt-series super-resolution
        opt.crop = None
        if opt.tilt_series > 0:
            # Scale factor
            scale = opt.tilt_series_in // opt.tilt_series_out
            opt.scale = (scale, 1, 1)
            opt.outputsz = tuple(np.array(opt.fov) * np.array(opt.scale))
            if opt.tilt_series_crop is not None:
                opt.crop = [o/float(f) for f,o in zip(opt.outputsz, opt.tilt_series_crop)]
                # Update output size
                opt.outputsz = tuple(opt.tilt_series_crop)
        else:
            # Output cropping
            diff = np.array(opt.fov) - np.array(opt.outputsz)
            assert all(diff >= 0)
            if any(diff > 0):
                opt.crop = [o/float(f) for f,o in zip(opt.fov, opt.outputsz)]

        # Super-resolution inference
        if opt.sr_mode:
            assert opt.tilt_series == 0, "sr_mode and tilt_series are mutually exclusive"
            assert opt.sr_scale_z > 1
            assert opt.inputsz[0] % opt.sr_scale_z == 0, \
                f"Input Z ({opt.inputsz[0]}) must be divisible by sr_scale_z ({opt.sr_scale_z})"
            if opt.sr_input_iso:
                # Iso input: scanner reads full iso patches, avg-downsample
                # + zero-pad happens per-patch in forward pass. No scale needed
                # (output is same resolution as input).
                pass
            else:
                # Aniso input: scanner reads Z/sr_scale_z patches, zero-pad
                # in forward pass. Scale maps output to iso coordinates.
                opt.scale = (opt.sr_scale_z, 1, 1)
                opt.sr_iso_inputsz = opt.inputsz
                aniso_z = opt.inputsz[0] // opt.sr_scale_z
                opt.inputsz = (aniso_z, opt.inputsz[1], opt.inputsz[2])
                opt.in_spec = dict(input=(1,) + opt.inputsz)

        # Per-class out_spec entries (driven by class registry).
        from deepem.data.classes import REGISTRY as CLASS_REGISTRY
        for flag_name, spec in CLASS_REGISTRY.items():
            if flag_name in ('vec', 'long'):
                continue  # Integer-valued in test; handled below.
            if getattr(opt, flag_name, False):
                opt.out_spec[spec.internal_name] = (spec.resolve_channels(opt),) + opt.outputsz

        # Integer-valued channel flags (legacy CLI shape).
        if opt.vec:
            opt.out_spec['embedding'] = (opt.vec,) + opt.outputsz
        if opt.long:
            opt.out_spec['long_range'] = (opt.long,) + opt.outputsz
        if opt.aff_deprecated:
            opt.out_spec['affinity'] = (opt.aff_deprecated,) + opt.outputsz

        # mito_to_cell also modifies in_spec.
        if opt.mito_to_cell:
            opt.in_spec['input_mitochondria'] = (1,) + opt.inputsz

        # --sem: shorthand for enabling all combined semantic-segmentation heads.
        if opt.sem:
            opt.out_spec['soma'] = (1,) + opt.outputsz
            opt.out_spec['axon'] = (1,) + opt.outputsz
            opt.out_spec['dendrite'] = (1,) + opt.outputsz
            opt.out_spec['glia'] = (1,) + opt.outputsz
            opt.out_spec['bvessel'] = (1,) + opt.outputsz

        # Semantic segmentation
        if opt.semantic:
            required_keys = ['soma', 'axon', 'dendrite', 'glia', 'blood_vessel']

            # Ensure all required keys are present in the opt.out_spec
            assert all(key in opt.out_spec for key in required_keys)

            # Use OrderedDict to maintain order of required keys followed by other keys
            out_spec_new = OrderedDict((key, opt.out_spec[key]) for key in required_keys)

            # Add remaining keys to out_spec_new
            out_spec_new.update((key, opt.out_spec[key]) for key in opt.out_spec if key not in required_keys)

            # Convert back to standard dict if necessary
            opt.out_spec = dict(out_spec_new)

        assert(len(opt.out_spec) > 0)

        # Scan spec
        opt.scan_spec = dict()
        if opt.vec:
            dim = opt.vec
            if opt.vec_to == 'aff':
                dim = len(opt.edges)
            if opt.vec_to == 'pca':
                dim = 3
            opt.scan_spec['embedding'] = (dim,) + opt.outputsz
        if opt.aff:
            opt.scan_spec['affinity'] = (3,) + opt.outputsz
        if opt.aff_overseg:
            opt.scan_spec['affinity_overseg'] = (3,) + opt.outputsz
        if opt.aff_deprecated:
            opt.scan_spec['affinity'] = (3,) + opt.outputsz
        if opt.bdr:
            opt.scan_spec['boundary'] = (1,) + opt.outputsz
        if opt.syn:
            opt.scan_spec['synapse'] = (1,) + opt.outputsz
        if opt.psd:
            opt.scan_spec['synapse'] = (1,) + opt.outputsz
        if opt.mit:
            opt.scan_spec['mitochondria'] = (1,) + opt.outputsz
        if opt.mito_to_cell:
            opt.scan_spec['mitochondria_to_cell'] = (1,) + opt.outputsz
        if opt.mye:
            opt.scan_spec['myelin'] = (1,) + opt.outputsz
        if opt.blv:
            opt.scan_spec['blood_vessel'] = (opt.blv_num_channels,) + opt.outputsz
        if opt.glia:
            opt.scan_spec['glia'] = (1,) + opt.outputsz
        if opt.sem:
            opt.scan_spec['soma'] = (1,) + opt.outputsz
            opt.scan_spec['axon'] = (1,) + opt.outputsz
            opt.scan_spec['dendrite'] = (1,) + opt.outputsz
            opt.scan_spec['glia'] = (1,) + opt.outputsz
            opt.scan_spec['bvessel'] = (1,) + opt.outputsz
        if opt.img:
            opt.scan_spec['image'] = (1,) + opt.outputsz
        if opt.dend:
            opt.scan_spec['dendrite'] = (1,) + opt.outputsz
        if opt.axon:
            opt.scan_spec['axon'] = (1,) + opt.outputsz
        if opt.soma:
            opt.scan_spec['soma'] = (1,) + opt.outputsz
        if opt.nucl:
            opt.scan_spec['nucleus'] = (1,) + opt.outputsz
        if opt.ecs:
            opt.scan_spec['extracellular_space'] = (1,) + opt.outputsz
        if opt.other:
            opt.scan_spec['other_class'] = (1,) + opt.outputsz

        # Semantic segmentation
        if opt.semantic:
            required_keys = ['soma', 'axon', 'dendrite', 'glia', 'blood_vessel']

            # Ensure all required keys are present in the opt.scan_spec
            assert all(key in opt.scan_spec for key in required_keys)

            # Use OrderedDict to maintain order of required keys followed by other keys
            scan_spec_new = OrderedDict((key, opt.scan_spec[key]) for key in required_keys)

            # Add remaining keys to scan_spec_new
            scan_spec_new.update((key, opt.scan_spec[key]) for key in opt.scan_spec if key not in required_keys)

            # Convert back to standard dict if necessary
            opt.scan_spec = dict(scan_spec_new)

        # Test-time augmentation
        if opt.test_aug16:
            opt.test_aug = list(range(16))

        if opt.test_aug64:
            opt.test_aug = list(range(64))

        # Overlap & stride
        if opt.stride is None:
            # infer stride from overlap
            opt.overlap = self.get_overlap(opt.outputsz, opt.overlap)
            opt.stride = tuple(int(f-o) for f,o in zip(opt.outputsz, opt.overlap))
        else:
            # infer overlap from stride
            stride = np.array(opt.stride) * np.array(opt.scale)
            opt.overlap = tuple(int(f-s) for f,s in zip(opt.outputsz, stride))

        # SR aniso: convert stride from iso output space to aniso input space
        if opt.sr_mode and not opt.sr_input_iso:
            opt.stride = tuple(
                int(s // sc) for s, sc in zip(opt.stride, opt.scale)
            )

        opt.scan_params = dict(stride=opt.stride, blend=opt.blend, scale=opt.scale)

        # Output tagging
        if opt.tags is not None:
            for k in opt.tags:
                assert k in opt.scan_spec

        # Validate mutually exclusive cloud-volume inputs
        if opt.gs_input and opt.gs_inputs:
            raise ValueError("Cannot use both --gs_input and --gs_inputs. Please use only one.")

        # Handle cloud-volume inputs
        if opt.gs_input:
            # Backward compatibility - convert single input to dict format
            opt.gs_inputs = {'input': opt.gs_input}
            opt.in_mips = {'input': opt.in_mip}
            opt.coord_mips = {'input': opt.coord_mip}
        elif opt.gs_inputs:
            if 'input' not in opt.gs_inputs:
                raise KeyError("Input key must be present in --gs_inputs")
            opt.gs_input = opt.gs_inputs['input']
            # Set default in_mip and coord_mip for all inputs if not specified
            for key in opt.gs_inputs:
                if key not in opt.in_mips:
                    opt.in_mips[key] = opt.in_mip
                if key not in opt.coord_mips:
                    opt.coord_mips[key] = opt.coord_mip
        opt.gs_input_masks = {'input': opt.gs_input_mask} if opt.gs_input_mask else {}
        opt.gs_input_norms = {'input': opt.gs_input_norm} if opt.gs_input_norm is not None else {}

        args = vars(opt)
        print('------------ Options -------------')
        for k, v in args.items():
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        self.opt = opt
        return self.opt

    def get_overlap(self, fov, overlap):
        assert len(fov) == 3
        assert len(overlap) == 3
        overlap_filter = lambda f,o: math.floor(f*o) if o > 0 and o < 1 else o
        return tuple(int(overlap_filter(f,o)) for f,o in zip(fov,overlap))
