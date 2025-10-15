import os
import time

from collections import defaultdict
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import samwise

from deepem.train.logger import Logger
from deepem.train.wandb_logger import WandbLogger
from deepem.train.option import Options
from deepem.train.utils import *

from deepem.utils.onnx_utils import export_onnx
from deepem.utils.torch_utils import revert_sync_batchnorm


def setup_distributed(backend='nccl'):
    dist.init_process_group(backend=backend)


def cleanup_distributed():
    dist.destroy_process_group()


def train(opt):
    # Model
    if opt.parallel == "DDP":
        # Make sure samewise finished syncing files
        dist.barrier()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        model = load_model(opt)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(local_rank)  # Move model to the corresponding GPU
        model = DDP(model, device_ids=[local_rank])  # Wrap model with DDP
    else:
        local_rank = None
        model = load_model(opt)

    # Optimizer
    trainable = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = load_optimizer(opt, trainable)

    # Data loaders
    train_loader, val_loader = load_data(opt, local_rank)

    # Initial checkpoint
    if opt.parallel == "DDP":
        if dist.get_rank() == 0:
            model = revert_sync_batchnorm(model)
            save_chkpt(model.module, opt.model_dir, opt.chkpt_num, optimizer)
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    else:
        save_chkpt(model, opt.model_dir, opt.chkpt_num, optimizer)

    # Mixed-precision training
    if opt.mixed_precision:
        scaler = torch.cuda.amp.GradScaler()

    # Training loop
    print("========== BEGIN TRAINING LOOP ==========")
    with Logger(opt) as logger, WandbLogger(opt) as wandb_logger:

        # Timer
        t0 = time.time()

        grad_accum_steps = opt.grad_accum_steps
        accum_losses = defaultdict(float)
        accum_nmasks = defaultdict(float)

        for i in range(opt.chkpt_num, opt.max_iter):
            # Load training samples.
            sample = train_loader()

            # Zero out gradients
            for param in model.parameters():
                param.grad = None

            # Optimizer step
            if opt.mixed_precision:
                with torch.cuda.amp.autocast():
                    losses, nmasks, preds = forward(model, sample, opt)
                    total_loss = sum([w*losses[k] for k, w in opt.loss_weight.items()])
                # Backward passes under autocast are not recommended.
                scaler.scale(total_loss / grad_accum_steps).backward()
            else:
                losses, nmasks, preds = forward(model, sample, opt)
                total_loss = sum([w * losses[k] for k, w in opt.loss_weight.items()])
                (total_loss / grad_accum_steps).backward()

            # Accumulate metrics for logging
            with torch.no_grad():
                for k, v in losses.items():
                    accum_losses[k] += v
                for k, v in nmasks.items():
                    accum_nmasks[k] += v

            if ((i - opt.chkpt_num) + 1) % grad_accum_steps != 0:
                continue

            # --- From here on, code only runs on optimizer step ---
            if opt.mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # Average accumulated losses
            avg_losses = {k: v / grad_accum_steps for k, v in accum_losses.items()}
            avg_nmasks = {k: v / grad_accum_steps for k, v in accum_nmasks.items()}

            if opt.mixed_precision:
                avg_losses = {k: v.float() for k, v in avg_losses.items()}
                avg_nmasks = {k: v.float() for k, v in avg_nmasks.items()}
                preds = {k: v.float() for k, v in preds.items()}


            # Elapsed time
            elapsed = time.time() - t0

            # Record keeping
            logger.record("train", avg_losses, avg_nmasks, elapsed=elapsed)

            # Reset accumulators
            accum_losses = defaultdict(float)
            accum_nmasks = defaultdict(float)

            # Log & display averaged stats.
            if (i+1) % opt.avgs_intv == 0 or i < opt.warm_up:
                stats = logger.check('train', i+1)
                wandb_logger.log_metrics('train', i+1, stats)

            # Image logging
            if (i+1) % opt.imgs_intv == 0:
                wandb_logger.log_images('train', i+1, preds, sample)

            # Evaluation loop
            if (i+1) % opt.eval_intv == 0:
                if opt.parallel == "DDP":
                    if dist.get_rank() == 0:
                        model = revert_sync_batchnorm(model)
                        eval_loop(i+1, model, val_loader, opt, logger, wandb_logger)
                        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                else:
                    eval_loop(i+1, model, val_loader, opt, logger, wandb_logger)

            # Frontier checkpoint (overwrites previous frontier)
            if opt.chkpt_sync_intv is not None and (i+1) % opt.chkpt_sync_intv == 0:
                if opt.parallel == "DDP":
                    if dist.get_rank() == 0:
                        model = revert_sync_batchnorm(model)
                        save_frontier_chkpt(model.module, opt.model_dir, i+1, optimizer)
                        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                else:
                    save_frontier_chkpt(model, opt.model_dir, i+1, optimizer)

            # Model checkpoint
            if (i+1) % opt.chkpt_intv == 0:
                if opt.parallel == "DDP":
                    if dist.get_rank() == 0:
                        model = revert_sync_batchnorm(model)
                        save_chkpt(model.module, opt.model_dir, i+1, optimizer)
                        if opt.export_onnx:
                            export_onnx(opt, i+1)
                        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                else:
                    save_chkpt(model, opt.model_dir, i+1, optimizer)
                    if opt.export_onnx:
                        export_onnx(opt, i+1)

            # Reset timer.
            t0 = time.time()


def eval_loop(iter_num, model, data_loader, opt, logger, wandb_logger):
    if not opt.no_eval:
        model.eval()

    # Evaluation loop
    print("---------- BEGIN EVALUATION LOOP ----------")
    with torch.no_grad():
        t0 = time.time()
        for i in range(opt.eval_iter):
            sample = data_loader()
            if opt.mixed_precision:
                with torch.cuda.amp.autocast():
                    losses, nmasks, preds = forward(model, sample, opt)
                losses = {k: v.float() for k, v in losses.items()}
                nmasks = {k: v.float() for k, v in nmasks.items()}
                preds  = {k: v.float() for k, v in preds.items()}
            else:
                losses, nmasks, preds = forward(model, sample, opt)
            elapsed = time.time() - t0

            # Record keeping
            logger.record('test', losses, nmasks, elapsed=elapsed)

            # Restart timer.
            t0 = time.time()

    # Log & display averaged stats.
    stats = logger.check('test', iter_num)
    wandb_logger.log_metrics('test', iter_num, stats)

    # Image logging
    if iter_num % opt.imgs_intv == 0:
        wandb_logger.log_images('test', iter_num, preds, sample)
    print("-------------------------------------------")

    model.train()


if __name__ == "__main__":

    # Options
    opt = Options().parse()

    if opt.parallel == "DDP":
        setup_distributed()
    # GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(opt.gpu_ids)

    # Make directories.
    if opt.parallel == "DDP" and dist.get_rank() == 0:
        if not os.path.isdir(opt.exp_dir):
            os.makedirs(opt.exp_dir)
        if not os.path.isdir(opt.log_dir):
            os.makedirs(opt.log_dir)
        if not os.path.isdir(opt.model_dir):
            os.makedirs(opt.model_dir)

    # cuDNN auto-tuning
    torch.backends.cudnn.benchmark = not opt.no_autotune

    # Run experiment.
    print(f"Running experiment: {opt.exp_name}")
    if opt.samwise_map is None:
        train(opt)
    elif opt.parallel != "DDP" or dist.get_rank() == 0:
        def f():
            train(opt)
        samwise.run(f, opt.samwise_map, period=opt.samwise_period)
    else:
        if dist.get_rank() % len(opt.gpu_ids) == 0:
            samwise.storage.initdirs(opt.samwise_map)
        train(opt)

    if opt.parallel == "DDP":
        cleanup_distributed()
