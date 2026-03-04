import argparse
import json
import os
import numpy as np
import samwise

from deepem.utils.py_utils import vec3, vec3f


def parse_class_weight(value):
    """Parse class weight as either scalar or comma-separated list.

    Args:
        value: String value from command line (e.g., "0.9" or "0.9,0.9,0.75")

    Returns:
        float or list of 3 floats

    Raises:
        ValueError: If format is invalid or list doesn't have exactly 3 values
    """
    if ',' in value:
        weights = [float(w.strip()) for w in value.split(',')]
        if len(weights) != 3:
            raise ValueError(
                f"Directional class weights must have exactly 3 values (x,y,z), "
                f"got {len(weights)}"
            )
        return weights
    else:
        return float(value)


class Options(object):
    """
    Training options.
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--exp_name', required=True)
        self.parser.add_argument('--model',    required=True)
        self.parser.add_argument('--sampler',  required=True)
        self.parser.add_argument('--data',     default=None)
        self.parser.add_argument('--augment',  default=None)
        self.parser.add_argument('--modifier', default=None)
        self.parser.add_argument('--modifier_kwargs', type=json.loads, default={})

        # zettasets
        self.parser.add_argument('--zettaset_path', type=str, default=[], nargs='+')
        self.parser.add_argument('--zettaset_specs', type=json.loads, default={})
        self.parser.add_argument('--zettaset_lookup', type=json.loads, default=None)
        self.parser.add_argument('--zettaset_padding', type=vec3, default=(0, 0, 0))
        self.parser.add_argument('--zettaset_padding_spec', type=json.loads, default={})
        self.parser.add_argument('--zettaset_resolution', type=vec3f, default=None)
        self.parser.add_argument('--zettaset_no_mask', action='store_true')
        self.parser.add_argument('--zettaset_share_mask', type=str, default=None)

        # file synchronization for spot/preemptible training
        self.parser.add_argument('--samwise_map', nargs='*', default=None)
        self.parser.add_argument('--samwise_period', type=int, default=600)        

        # cuDNN auto-tuning
        self.parser.add_argument('--no_autotune', action='store_true')

        # Training/validation sets
        self.parser.add_argument('--train_ids', type=str, default=[], nargs='+')
        self.parser.add_argument('--train_prob', type=float, default=None, nargs='+')
        self.parser.add_argument('--val_ids', type=str, default=[], nargs='+')
        self.parser.add_argument('--val_prob', type=float, default=None, nargs='+')

        # Training
        self.parser.add_argument('--max_iter', type=int, default=1000000)
        self.parser.add_argument('--batch_size', type=int, default=1)
        self.parser.add_argument('--num_workers', type=int, default=1)
        self.parser.add_argument('--gpu_ids', type=str, default=['0'], nargs='+')
        self.parser.add_argument('--eval_intv', type=int, default=1000)
        self.parser.add_argument('--eval_iter', type=int, default=100)
        self.parser.add_argument('--avgs_intv', type=int, default=100)
        self.parser.add_argument('--imgs_intv', type=int, default=1000)
        self.parser.add_argument('--warm_up', type=int, default=100)
        self.parser.add_argument('--chkpt_intv', type=int, default=10000)
        self.parser.add_argument('--chkpt_sync_intv', type=int, default=None)
        self.parser.add_argument('--chkpt_num', type=int, default=0)
        self.parser.add_argument('--no_eval', action='store_true')
        self.parser.add_argument('--pretrain', default=None)

        # WandB logging
        self.parser.add_argument('--wandb_pad_output', action='store_true')

        # Loss
        self.parser.add_argument('--loss', default='BCELoss')
        self.parser.add_argument('--no_split_boundary', action='store_true')
        self.parser.add_argument('--size_average', action='store_true')
        self.parser.add_argument('--margin0', type=float, default=0)
        self.parser.add_argument('--margin1', type=float, default=0)
        self.parser.add_argument('--inverse', action='store_true')
        self.parser.add_argument('--class_balancing', action='store_true')
        self.parser.add_argument('--class_weight0', type=parse_class_weight, default=None)
        self.parser.add_argument('--class_weight1', type=parse_class_weight, default=None)
        self.parser.add_argument('--default_aux', action='store_true')

        # Mean-based loss
        self.parser.add_argument('--metric_loss', default='MeanLoss')
        self.parser.add_argument('--scale_init', type=float, default=1.0)
        self.parser.add_argument('--alpha', type=float, default=1.0)
        self.parser.add_argument('--beta', type=float, default=1.0)
        self.parser.add_argument('--gamma', type=float, default=0.001)
        self.parser.add_argument('--delta_v', type=float, default=0.0)
        self.parser.add_argument('--delta_d', type=float, default=1.5)
        self.parser.add_argument('--recompute_ext', action='store_true')
        self.parser.add_argument('--no_mask_background', action='store_true')
        self.parser.add_argument('--loss_scale_factor', type=vec3f, default=None)

        # Optimizer
        self.parser.add_argument('--optim', default='Adam')
        self.parser.add_argument('--lr', type=float, default=0.001)

        # Optimizer: Adam
        self.parser.add_argument('--betas', type=float, default=[0.9,0.999], nargs='+')
        self.parser.add_argument('--eps', type=float, default=1e-08)
        self.parser.add_argument('--amsgrad', action='store_true')

        # Optimizer: SGD
        self.parser.add_argument('--momentum', type=float, default=0.9)

        # Model architecture
        self.parser.add_argument('--inputsz', type=vec3, default=None)
        self.parser.add_argument('--outputsz', type=vec3, default=None)
        self.parser.add_argument('--fov', type=vec3, default=(20,256,256))
        self.parser.add_argument('--depth', type=int, default=4)
        self.parser.add_argument('--width', type=int, default=None, nargs='+')
        self.parser.add_argument('--group', type=int, default=0)
        self.parser.add_argument('--group_eps', type=float, default=1e-5)
        self.parser.add_argument('--act', default='ReLU')
        self.parser.add_argument('--updown_scale_factor', type=vec3f, default=None)

        # Data augmentation
        self.parser.add_argument('--recompute', type=str, default=[], nargs='+')
        self.parser.add_argument('--border', type=str, default=[], nargs='+')
        self.parser.add_argument('--flip', action='store_true')
        self.parser.add_argument('--grayscale', action='store_true')
        self.parser.add_argument('--warping', action='store_true')
        self.parser.add_argument('--misalign', type=int, default=0)
        self.parser.add_argument('--interp', action='store_true')
        self.parser.add_argument('--missing', type=int, default=0)
        self.parser.add_argument('--blur', type=int, default=0)
        self.parser.add_argument('--box', default=None)
        self.parser.add_argument('--mip', type=int, default=0)
        self.parser.add_argument('--lost', action='store_true')
        self.parser.add_argument('--random', action='store_true')
        self.parser.add_argument('--noise', action='store_true')
        self.parser.add_argument('--noise_min', type=float, default=0.01)
        self.parser.add_argument('--noise_max', type=float, default=0.1)
        self.parser.add_argument('--noise_per_channel', action='store_true')
        self.parser.add_argument('--section_gap', type=int, default=0)
        self.parser.add_argument('--mask_section_gap', action='store_true')
        self.parser.add_argument('--degradation_skip', type=float, default=0.1)

        # Tilt-series electron tomography
        self.parser.add_argument('--tilt_series', type=int, default=0)
        self.parser.add_argument('--tilt_series_in', type=int, default=12)
        self.parser.add_argument('--tilt_series_out', type=int, default=4)
        self.parser.add_argument('--tilt_series_crop', type=vec3, default=None)

        # Super-resolution mode
        self.parser.add_argument('--sr_mode', action='store_true',
                                 help='Enable super-resolution mode for mixed iso/aniso training')
        self.parser.add_argument('--sr_scale_z', type=int, default=5,
                                 help='Z upsampling factor (default: 5 for 40nm:8nm ratio)')

        # Long-range affinity
        self.parser.add_argument('--long', type=float, default=0)
        self.parser.add_argument('--edges', type=vec3, default=[], nargs='+')

        # Multiclass detection
        self.parser.add_argument('--aff', type=float, default=0)  # Affinity
        self.parser.add_argument('--bdr', type=float, default=0)  # Boundary
        self.parser.add_argument('--syn', type=float, default=0)  # Synapse
        self.parser.add_argument('--psd', type=float, default=0)  # Synapse
        self.parser.add_argument('--mit', type=float, default=0)  # Mitochondria
        self.parser.add_argument('--mye', type=float, default=0)  # Myelin
        self.parser.add_argument('--fld', type=float, default=0)  # Fold
        self.parser.add_argument('--blv', type=float, default=0)  # Blood vessel
        self.parser.add_argument('--blv_num_channels', type=int, default=1)
        self.parser.add_argument('--glia', type=float, default=0) # Glia
        self.parser.add_argument('--glia_mask', action='store_true')
        self.parser.add_argument('--img', type=float, default=0)  # Image
        self.parser.add_argument('--mito_to_cell', type=float, default=0)  # Mito to cell
        self.parser.add_argument('--mito_to_cell_mode', type=str, default='random')  # Mito to cell mode
        self.parser.add_argument('--merge_classes', type=str, default=[], nargs='+')  # for onnx export

        # Semantic segmentation
        self.parser.add_argument('--sem', action='store_true')
        self.parser.add_argument('--dend', type=float, default=0)  # Dendrite
        self.parser.add_argument('--axon', type=float, default=0)  # Axon
        self.parser.add_argument('--soma', type=float, default=0)  # Soma
        self.parser.add_argument('--nucl', type=float, default=0)  # Nucleus
        self.parser.add_argument('--ecs',  type=float, default=0)  # Extracellular space
        self.parser.add_argument('--other', type=float, default=0) # Other class

        # Metric learning
        self.parser.add_argument('--vec', type=float, default=0)
        self.parser.add_argument('--embed_dim', type=int, default=12)

        # Mitochondria embedding
        self.parser.add_argument('--mito_emb', type=float, default=0)
        self.parser.add_argument('--mito_emb_dim', type=int, default=6)

        # Test training
        self.parser.add_argument('--test', action='store_true')

        # Mixed-precision training
        self.parser.add_argument('--mixed_precision', type=str, default=None, choices=['fp16', 'bf16'])

        # Export to ONNX
        self.parser.add_argument('--export_onnx', action='store_true')
        self.parser.add_argument('--opset_version', type=int, default=10)

        self.parser.add_argument('--parallel', type=str, choices=["DDP", "DP", ], default=None)

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()

        if not opt.parallel:
            if "RANK" in os.environ or "LOCAL_RANK" in os.environ:
                opt.parallel = "DDP"

        # Directories
        if opt.exp_name.split('/')[0] == 'experiments':
            opt.exp_dir = opt.exp_name
        else:
            opt.exp_dir = f"experiments/{opt.exp_name}"
        if opt.test:
            opt.exp_dir = 'test/' + opt.exp_dir
        opt.log_dir = os.path.join(opt.exp_dir, 'logs')
        opt.model_dir = os.path.join(opt.exp_dir, 'models')

        # Samwise synchronization
        if opt.samwise_map is not None:
            opt.samwise_map = samwise.parse.parsemap(opt.samwise_map)

        # Training/validation sets
        if (not opt.train_ids) or (not opt.val_ids):
            raise ValueError("Train/validation IDs unspecified")
        if opt.train_prob:
            if len(opt.train_ids) != len(opt.train_prob):
                error_message = (
                    "The lengths of 'train_ids' and 'train_prob' must be the same. "
                    f"train_ids: {opt.train_ids}, train_prob: {opt.train_prob}"
                )
                raise ValueError(error_message)
        if opt.val_prob:
            assert len(opt.val_ids) == len(opt.val_prob)

        args = vars(opt)

        # Loss
        loss_keys = ['size_average','margin0','margin1','inverse']
        opt.loss_params = {k: args[k] for k in loss_keys}

        # Metirc learning
        assert opt.metric_loss in ['MeanLoss']
        opt.metric_params = dict()
        opt.metric_params['alpha'] = opt.alpha
        opt.metric_params['beta'] = opt.beta
        opt.metric_params['gamma'] = opt.gamma
        opt.metric_params['delta_v'] = opt.delta_v
        opt.metric_params['delta_d'] = opt.delta_d
        opt.metric_params['recompute_ext'] = opt.recompute_ext
        opt.metric_params['mask_background'] = not opt.no_mask_background
        opt.metric_params['loss_scale_factor'] = opt.loss_scale_factor

        # Optimizer
        if opt.optim in ['Adam', 'AdamW']:
            optim_keys = ['lr','betas','eps','amsgrad']
        elif opt.optim == 'SGD':
            optim_keys = ['lr','momentum']
        else:
            optim_keys = ['lr']
        opt.optim_params = {k: args[k] for k in optim_keys}

        # Data augmentation
        aug_keys = ['recompute', 'border', 'flip','grayscale','warping','misalign',
                    'interp','missing','blur','box','mip','lost','random',
                    'section_gap', 'mask_section_gap', 'degradation_skip']
        opt.aug_params = {k: args[k] for k in aug_keys}

        # Noise
        if opt.noise:
            opt.aug_params['noise'] = (opt.noise_min, opt.noise_max,
                                       opt.noise_per_channel)

        # Model
        opt.fov = tuple(opt.fov)
        opt.inputsz = opt.fov if opt.inputsz is None else tuple(opt.inputsz)
        opt.outputsz = opt.fov if opt.outputsz is None else tuple(opt.outputsz)
        opt.in_spec = dict(input=(1,) + opt.inputsz)
        opt.out_spec = dict()
        opt.loss_weight = dict()

        # Output cropping
        opt.crop = None
        if opt.tilt_series > 0:
            # Input size
            factor = (opt.tilt_series_in, 1, 1)
            opt.inputsz = tuple(np.array(opt.fov) // np.array(factor))
            opt.in_spec = dict(input=(opt.tilt_series,) + opt.inputsz)
            # Output size
            scale = (opt.tilt_series_in // opt.tilt_series_out, 1, 1)
            opt.outputsz = tuple(np.array(opt.inputsz) * np.array(scale))
            if opt.tilt_series_crop:
                opt.crop = [o/float(f) for f,o in zip(opt.outputsz, opt.tilt_series_crop)]
                # Update output size
                opt.outputsz = tuple(opt.tilt_series_crop)
        else:
            diff = np.array(opt.fov) - np.array(opt.outputsz)
            assert all(diff >= 0)
            if any(diff > 0):
                opt.crop = [o/float(f) for f,o in zip(opt.fov, opt.outputsz)]

        # Tilt-series electron tomography
        opt.aug_params['tilt_series'] = (opt.tilt_series,
                                         opt.tilt_series_in,
                                         opt.tilt_series_out)
        opt.aug_params['tilt_series_crop'] = opt.crop

        # Super-resolution augmentation params
        opt.aug_params['sr_mode'] = opt.sr_mode
        opt.aug_params['sr_scale_z'] = opt.sr_scale_z

        # Multiclass detection
        class_keys = list()
        class_dict = {
            'aff':  ('affinity', 3),
            'long': ('long_range', len(opt.edges)),
            'bdr':  ('boundary', 1),
            'syn':  ('synapse', 1),
            'psd':  ('synapse', 1),
            'mit':  ('mitochondria', 1),
            'mye':  ('myelin', 1),
            'fld':  ('fold', 1),
            'blv':  ('blood_vessel', opt.blv_num_channels),
            'glia': ('glia', 1),
            'soma': ('soma', 1),
            'img':  ('image', 1),
            'vec':  ('embedding', opt.embed_dim),
            'dend': ('dendrite', 1),
            'axon': ('axon', 1),
            'nucl': ('nucleus', 1),
            'ecs':  ('extracellular_space', 1),
            'other':  ('other_class', 1),
            'mito_to_cell': ('mitochondria_to_cell', 1),
            'mito_emb': ('mitochondria_embedding', opt.mito_emb_dim),
        }

        semantic_mapping = {
            'dendrite': 1,
            'axon': 2,
            'soma': 3,
            'nucleus': 4,
            'glia': 5,
            'extracellular_space': 6,
            'blood_vessel': 7,
            'other_class': 10,
        }

        requires_binarize = [
            "synapse",
            "mitochondria",
            "myelin",
            "fold",
            "glia",
            "soma",
        ]

        if opt.blv_num_channels == 1:
            requires_binarize.append("blood_vessel")

        if opt.sem:
            requires_binarize = [x for x in requires_binarize if x not in semantic_mapping]

        # Test training
        if opt.test:
            opt.eval_intv = 100
            opt.eval_iter = 10
            opt.avgs_intv = 10
            opt.imgs_intv = 100

        for k, v in class_dict.items():
            loss_w = args[k]
            if loss_w > 0:
                output_name, num_channels = v
                assert num_channels > 0
                opt.out_spec[output_name] = (num_channels,) + opt.outputsz
                opt.loss_weight[output_name] = loss_w
                class_keys.append(k)

        # Mito-to-cell assignment hacks
        if opt.mito_to_cell > 0:
            opt.in_spec = {
                'input': (1,) + opt.inputsz,
                'input_mitochondria': (1,) + opt.inputsz,
            }
            opt.out_spec['mitochondria_to_cell'] = (1,) + opt.inputsz

        assert len(opt.out_spec) > 0
        assert len(opt.out_spec) == len(opt.loss_weight) == len(class_keys)
        opt.data_params = dict(
            class_keys=class_keys,
            glia_mask=opt.glia_mask,
            zettaset_lookup=opt.zettaset_lookup,
            zettaset_padding=opt.zettaset_padding,
            zettaset_padding_spec=opt.zettaset_padding_spec,
            zettaset_resolution=opt.zettaset_resolution,
            zettaset_mask=not opt.zettaset_no_mask,
            requires_binarize=requires_binarize,
            zettaset_share_mask=opt.zettaset_share_mask,
            semantic_mapping=semantic_mapping if opt.sem else {},
        )

        # Sampler
        opt.sampler_params = {
            'mode': opt.mito_to_cell_mode,
            'output_shape': opt.outputsz,
        }

        # ONNX
        opt.onnx = False

        # Print options
        args = vars(opt)
        print('------------ Options -------------')
        for k, v in args.items():
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        self.opt = opt
        return self.opt
