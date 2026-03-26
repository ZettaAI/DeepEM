import os
import glob

import torch
from torch.nn.parallel import data_parallel

import deepem.loss as loss
from deepem.train.data import Data
from deepem.train.model import Model, AmpModel
from deepem.loss.utils import BinaryWeightBalancer
from deepem.utils.py_utils import load_module


def get_criteria(opt):
    criteria = dict()

    # Class balancing
    balancer = BinaryWeightBalancer(
        weight0=opt.class_weight0,
        weight1=opt.class_weight1,
    ) if opt.class_balancing else None

    for k in opt.out_spec:
        if k == 'affinity' or k == 'long_range':
            if k == 'affinity':
                edges = [(0,0,1),(0,1,0),(1,0,0)]
            else:
                edges = list(opt.edges)
            assert len(edges) > 0
            params = dict(opt.loss_params)
            params['size_average'] = False
            criteria[k] = loss.AffinityLoss(edges,
                criterion=getattr(loss, opt.loss)(**params),
                split_boundary=not opt.no_split_boundary,
                size_average=opt.size_average,
                class_balancer=balancer,
            )
        elif k == 'embedding':
            criteria[k] = getattr(loss, opt.metric_loss)(**opt.metric_params)
        elif k == 'mitochondria_embedding':
            params = dict(opt.metric_params)
            params['mask_background'] = False
            criteria[k] = getattr(loss, opt.metric_loss)(**params)
        else:
            params = dict(opt.loss_params)

            if ('affinity' in opt.out_spec) or ('long_range' in opt.out_spec):
                # Auxiliary outputs use inverted weights
                # If directional weights provided, skip class balancing for auxiliary
                if isinstance(opt.class_weight0, list) or isinstance(opt.class_weight1, list):
                    balancer = None  # No class balancing for auxiliary with directional weights
                else:
                    balancer = BinaryWeightBalancer(
                        weight0=opt.class_weight1,
                        weight1=opt.class_weight0,
                    ) if opt.class_balancing else None
            params['class_balancer'] = balancer

            if opt.default_aux:
                params['margin0'] = 0
                params['margin1'] = 0
                params['inverse'] = False
                params['class_balancer'] = None

            criteria[k] = getattr(loss, 'BCELoss')(**params)
    return criteria


def load_model(opt):
    """Creates and loads a model based on options."""
    # Create base model
    mod = load_module("model", opt.model)
    model_class = AmpModel if opt.mixed_precision else Model
    model = model_class(mod.create_model(opt), get_criteria(opt), opt)

    # Load pretrained weights if specified
    if opt.pretrain:
        model.load(opt.pretrain)

    # Load checkpoint if specified
    if opt.chkpt_num != 0:
        model = load_chkpt(model, opt.model_dir, opt.chkpt_num)
        if opt.chkpt_num == -1:
            chkpt_num, is_frontier = latest_chkpt(opt.model_dir)
            if chkpt_num is not None:
                opt.chkpt_num = chkpt_num
                opt.loaded_from_frontier = is_frontier

    return model.train().cuda()


def load_optimizer(opt, trainable):
    # Create an optimizer.
    optimizer = getattr(torch.optim, opt.optim)(trainable, **opt.optim_params)

    if not opt.pretrain and opt.chkpt_num > 0:
        # Load optimizer state from the checkpoint we actually used
        if hasattr(opt, 'loaded_from_frontier') and opt.loaded_from_frontier:
            load_optimizer_state(optimizer, opt.model_dir, opt.chkpt_num, is_frontier=True)
        else:
            load_optimizer_state(optimizer, opt.model_dir, opt.chkpt_num, is_frontier=False)

    print(optimizer)
    return optimizer


def load_chkpt(model, fpath, chkpt_num):
    is_frontier = False

    if chkpt_num == -1:
        chkpt_num, is_frontier = latest_chkpt(fpath)
        if chkpt_num is None:
            print("No checkpoints found, starting fresh")
            return model

    fname = os.path.join(fpath, "model_frontier.chkpt" if is_frontier else f"model{chkpt_num}.chkpt")
    print(f"LOAD {'FRONTIER ' if is_frontier else ''}CHECKPOINT: {chkpt_num} iters.")
    model.load(fname)
    return model


def latest_chkpt(fpath):
    """Finds the best checkpoint by comparing regular and frontier checkpoints."""
    # Check frontier checkpoint first
    frontier_fname = os.path.join(fpath, "model_frontier.chkpt")
    frontier_iter = None
    if os.path.exists(frontier_fname):
        chkpt = torch.load(frontier_fname)
        frontier_iter = chkpt['iter']

    # Get regular checkpoint numbers (excluding frontier)
    modelfilenames = glob.glob(os.path.join(fpath, "model*.chkpt"))
    # Remove frontier checkpoint from the list
    if frontier_fname in modelfilenames:
        modelfilenames.remove(frontier_fname)

    def chkpt_num_from_filename(f):
        b = os.path.basename(f)
        filename = os.path.splitext(b)[0]
        return int(filename[5:])

    # Get regular checkpoint numbers
    chkpt_nums = [chkpt_num_from_filename(f) for f in modelfilenames]
    latest_regular = max(chkpt_nums) if len(chkpt_nums) > 0 else None

    # Compare and return the best checkpoint
    if frontier_iter is None and latest_regular is None:
        return None, False
    if frontier_iter is None:
        return latest_regular, False
    if latest_regular is None:
        return frontier_iter, True
    if frontier_iter > latest_regular:
        return frontier_iter, True
    return latest_regular, False


def save_chkpt(model, fpath, chkpt_num, optimizer):
    print(f"SAVE CHECKPOINT: {chkpt_num} iters.")
    fname = os.path.join(fpath, f"model{chkpt_num}.chkpt")
    state = {'iter': chkpt_num,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, fname)


def save_frontier_chkpt(model, fpath, chkpt_num, optimizer):
    """Save a frontier checkpoint that overwrites the previous frontier checkpoint."""
    print(f"SAVE FRONTIER CHECKPOINT: {chkpt_num} iters.")
    fname = os.path.join(fpath, "model_frontier.chkpt")
    state = {'iter': chkpt_num,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, fname)


def load_frontier_chkpt(model, fpath):
    """Load a frontier checkpoint."""
    fname = os.path.join(fpath, "model_frontier.chkpt")
    if os.path.exists(fname):
        print(f"LOAD FRONTIER CHECKPOINT.")
        model.load(fname)
        chkpt = torch.load(fname)
        return model, chkpt['iter']
    else:
        print(f"FRONTIER CHECKPOINT NOT FOUND: {fname}")
        return model, 0


def load_optimizer_state(optimizer, fpath, chkpt_num, is_frontier=False):
    """Load optimizer state from checkpoint."""
    if is_frontier:
        fname = os.path.join(fpath, "model_frontier.chkpt")
        if not os.path.exists(fname):
            return
        chkpt = torch.load(fname)
        iter_num = chkpt.get('iter', 0)
        print(f"LOAD FRONTIER OPTIM STATE: {iter_num} iters.")
    else:
        fname = os.path.join(fpath, f"model{chkpt_num}.chkpt")
        chkpt = torch.load(fname)
        print(f"LOAD OPTIM STATE: {chkpt_num} iters.")

    if 'optimizer' in chkpt:
        optimizer.load_state_dict(chkpt['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()


def load_data(opt, local_rank):
    # Extract val and train_prob from zettaset_specs
    spec_val_ids, spec_exclude_ids, supersede_val = _extract_val_from_specs(
        opt.zettaset_specs
    )
    # Drop CLI val_ids that are supersets with spec-level val (e.g., seuron default)
    if supersede_val:
        supersede_set = set(supersede_val)
        dropped = [v for v in opt.val_ids if v in supersede_set]
        if dropped:
            print(f"Dropping superset val_ids (superseded by spec): {dropped}")
        opt.val_ids = [v for v in opt.val_ids if v not in supersede_set]
    opt.val_ids = opt.val_ids + spec_val_ids

    data_ids = list(set().union(opt.train_ids, opt.val_ids))
    if opt.zettaset_specs:
        from deepem.data.dataset import multi_zettaset as mod
    else:
        from deepem.data.dataset import zettaset as mod
    data = mod.load_data(
        opt.zettaset_specs if opt.zettaset_specs else opt.zettaset_path,
        data_ids=data_ids,
        **opt.data_params
    )

    # Train (with exclusion filtering)
    exclude_ids = list(opt.train_exclude) + spec_exclude_ids
    if opt.exclude_val:
        exclude_ids.extend(opt.val_ids)
    train_data, excluded_ids = _filter_train_data(
        data, opt.train_ids, exclude_ids, opt.zettaset_specs
    )
    if opt.train_prob:
        prob = dict(zip(opt.train_ids, opt.train_prob))
        # Drop prob entries for fully-excluded train_ids
        prob = {k: v for k, v in prob.items() if k in train_data}
    else:
        prob = _extract_train_prob_from_specs(opt.zettaset_specs, train_data)
    train_loader = Data(opt, train_data, is_train=True, prob=prob, local_rank=local_rank)

    # Validation
    val_data = _build_val_data(data, opt.val_ids, opt.zettaset_specs)
    if opt.val_prob:
        prob = dict(zip(opt.val_ids, opt.val_prob))
    else:
        prob = None
    val_loader = Data(opt, val_data, is_train=False, prob=prob, local_rank=local_rank)

    return train_loader, val_loader


def _filter_train_data(data, train_ids, exclude_ids, zettaset_specs):
    """Filter excluded sample IDs out of training data.

    Handles both superset and sample-level exclusions:
    - ^hemibrain:lobula  -> removes that sample from the hemibrain superset
    - ^hemibrain         -> expands to all samples, removes them all

    Args:
        data: Loaded data dict from multi_zettaset.load_data.
        train_ids: List of training data IDs (inclusions only).
        exclude_ids: List of IDs to exclude (^ prefix already stripped).
        zettaset_specs: Zettaset specifications dict (may be empty/None).

    Returns:
        (filtered_data, excluded_samples): Filtered training data dict and
            the set of sample IDs that were actually excluded.
    """
    if not exclude_ids:
        return {k: data[k] for k in train_ids}, set()

    zettaset_specs = zettaset_specs or {}

    # Expand superset exclusions to sample-level
    expanded_excludes = set()
    for eid in exclude_ids:
        if eid in zettaset_specs and eid in data and isinstance(data[eid], dict):
            # Superset exclusion -> expand to all its samples
            expanded_excludes.update(data[eid].keys())
        else:
            expanded_excludes.add(eid)

    # Filter train data
    result = {}
    excluded_samples = set()
    for k in train_ids:
        if k in zettaset_specs and isinstance(data.get(k), dict):
            # Superset: filter out excluded children
            original = data[k]
            filtered = {sk: sv for sk, sv in original.items()
                        if sk not in expanded_excludes}
            newly_excluded = set(original.keys()) - set(filtered.keys())
            if newly_excluded:
                excluded_samples.update(newly_excluded)
                print(f"Excluded from '{k}': {sorted(newly_excluded)}")
            if filtered:
                result[k] = filtered
                print(f"Remaining in '{k}': {sorted(filtered.keys())}")
            else:
                print(f"WARNING: All samples excluded from superset '{k}'")
        else:
            # Sample ID: skip if excluded
            if k in expanded_excludes:
                excluded_samples.add(k)
                print(f"Excluded: {k}")
            else:
                result[k] = data[k]

    if excluded_samples:
        print(f"Total excluded from training: {len(excluded_samples)} sample(s)")

    return result, excluded_samples


def _extract_val_from_specs(zettaset_specs):
    """Extract val sample IDs and their exclusion IDs from zettaset_specs.

    Reads the "val" field from each zettaset spec and returns full sample IDs
    (zettaset_name:sample_name) for both val_ids and exclude_ids. Also returns
    superset names that have spec-level val, so that superset-level CLI val_ids
    can be replaced (e.g., seuron's default --val_ids hemibrain gets replaced
    by the spec-level hemibrain:lobula).

    Returns:
        (val_ids, exclude_ids, supersede_val): Three lists. supersede_val
            contains superset names whose CLI val_ids should be dropped.
    """
    if not zettaset_specs:
        return [], [], []
    val_ids = []
    exclude_ids = []
    supersede_val = []
    for name, spec in zettaset_specs.items():
        val_samples = spec.get("val", [])
        if not val_samples:
            continue
        supersede_val.append(name)
        val_train = spec.get("val_train", False)
        for sample_name in val_samples:
            full_id = f"{name}:{sample_name}"
            val_ids.append(full_id)
            if not val_train:
                exclude_ids.append(full_id)
    if val_ids:
        print(f"Val from zettaset_specs: {val_ids}")
    return val_ids, exclude_ids, supersede_val


def _extract_train_prob_from_specs(zettaset_specs, train_data):
    """Extract per-zettaset train_prob from zettaset_specs.

    Returns a prob dict if any spec has "train_prob", else None
    (letting DataProvider auto-compute from num_samples).
    """
    if not zettaset_specs:
        return None
    prob = {}
    for k in train_data:
        spec = zettaset_specs.get(k, {})
        if "train_prob" in spec:
            prob[k] = spec["train_prob"]
    if not prob:
        return None
    # If some but not all have train_prob, default missing ones to 1.0
    for k in train_data:
        if k not in prob:
            prob[k] = 1.0
    return prob


def _build_val_data(data, val_ids, zettaset_specs):
    """Build validation data dict, handling both superset and flat entries.

    Val IDs from zettaset_specs "val" field are loaded as individual samples
    but may only exist inside a superset's nested dict. This function extracts
    them from wherever they live in the data dict.
    """
    zettaset_specs = zettaset_specs or {}
    val_data = {}
    for k in val_ids:
        if k in data:
            # Direct match (individual sample or superset)
            val_data[k] = data[k]
        elif ':' in k:
            # Try extracting from parent superset
            zettaset_name = k.split(':')[0]
            if zettaset_name in data and isinstance(data[zettaset_name], dict):
                if k in data[zettaset_name]:
                    val_data[k] = data[zettaset_name][k]
                else:
                    raise KeyError(
                        f"Val sample '{k}' not found in superset '{zettaset_name}'"
                    )
            else:
                raise KeyError(f"Val sample '{k}' not found in loaded data")
        else:
            raise KeyError(f"Val ID '{k}' not found in loaded data")
    return val_data


def forward(model, sample, opt):
    # Forward pass
    if len(opt.gpu_ids) > 1 and opt.parallel == 'DP':
        losses, nmasks, preds = data_parallel(model, sample)
    else:
        losses, nmasks, preds = model(sample)

    # Average over minibatch
    losses = {k: v.mean() for k, v in losses.items()}
    nmasks = {k: v.mean() for k, v in nmasks.items()}

    return losses, nmasks, preds
