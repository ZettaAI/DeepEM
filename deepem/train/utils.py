import imp
import os
import glob

import torch
from torch.nn.parallel import data_parallel

import deepem.loss as loss
from deepem.train.data import Data
from deepem.train.model import Model, AmpModel
from deepem.loss.utils import BinaryWeightBalancer


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
        else:
            params = dict(opt.loss_params)

            if ('affinity' in opt.out_spec) or ('long_range' in opt.out_spec):
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
    # Create a model.
    mod = imp.load_source('model', opt.model)
    if opt.mixed_precision:
        model = AmpModel(mod.create_model(opt), get_criteria(opt), opt)
    else:
        model = Model(mod.create_model(opt), get_criteria(opt), opt)

    if opt.pretrain:
        model.load(opt.pretrain)
    if opt.chkpt_num == -1:
        opt.chkpt_num = latest_chkpt(opt.model_dir)
    if opt.chkpt_num > 0:
        model = load_chkpt(model, opt.model_dir, opt.chkpt_num)

    return model.train().cuda()


def load_optimizer(opt, trainable):
    # Create an optimizer.
    optimizer = getattr(torch.optim, opt.optim)(trainable, **opt.optim_params)

    if not opt.pretrain and opt.chkpt_num > 0:
        n = opt.chkpt_num
        fname = os.path.join(opt.model_dir, f"model{n}.chkpt")
        chkpt = torch.load(fname)
        if 'optimizer' in chkpt:
            print(f"LOAD OPTIM STATE: {n} iters.")
            optimizer.load_state_dict(chkpt['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

    print(optimizer)
    return optimizer


def load_chkpt(model, fpath, chkpt_num):
    if chkpt_num == -1:
        chkpt_num = latest_chkpt(fpath)

    print(f"LOAD CHECKPOINT: {chkpt_num} iters.")
    fname = os.path.join(fpath, f"model{chkpt_num}.chkpt")
    model.load(fname)
    return model


def latest_chkpt(fpath):
    """Finds the checkpoint with the largest iteration number."""
    modelfilenames = glob.glob(os.path.join(fpath, "model*.chkpt"))

    def chkpt_num_from_filename(f):
        b = os.path.basename(f)
        return int(os.path.splitext(b)[0][5:])

    chkpt_nums = [chkpt_num_from_filename(f) for f in modelfilenames]

    return max(chkpt_nums) if len(chkpt_nums) > 0 else 0


def save_chkpt(model, fpath, chkpt_num, optimizer):
    print(f"SAVE CHECKPOINT: {chkpt_num} iters.")
    fname = os.path.join(fpath, f"model{chkpt_num}.chkpt")
    state = {'iter': chkpt_num,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, fname)


def load_data(opt):
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

    # Train
    train_data = {k: data[k] for k in opt.train_ids}
    if opt.train_prob:
        prob = dict(zip(opt.train_ids, opt.train_prob))
    else:
        prob = None
    train_loader = Data(opt, train_data, is_train=True, prob=prob)

    # Validation
    val_data = {k: data[k] for k in opt.val_ids}
    if opt.val_prob:
        prob = dict(zip(opt.val_ids, opt.val_prob))
    else:
        prob = None
    val_loader = Data(opt, val_data, is_train=False, prob=prob)

    return train_loader, val_loader


def forward(model, sample, opt):
    # Forward pass
    if len(opt.gpu_ids) > 1:
        losses, nmasks, preds = data_parallel(model, sample)
    else:
        losses, nmasks, preds = model(sample)

    # Average over minibatch
    losses = {k: v.mean() for k, v in losses.items()}
    nmasks = {k: v.mean() for k, v in nmasks.items()}

    return losses, nmasks, preds
