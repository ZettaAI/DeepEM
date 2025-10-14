from __future__ import annotations

import argparse
import os
from types import TracebackType

import numpy as np
import torch
import torch.distributed as dist
from torchvision.utils import make_grid
import wandb

from deepem.loss.mean import vec2aff
from deepem.utils import py_utils, torch_utils


class WandbLogger:
    """
    Weight & Biases Logger.
    """

    def __init__(self, opt: argparse.Namespace):
        self.opt = opt
        self.in_spec = dict(opt.in_spec)
        self.out_spec = dict(opt.out_spec)
        self.pad_output = getattr(opt, 'wandb_pad_output', False)

        # Determine if this is a non-main DDP rank
        self._is_ddp_worker = (
            getattr(opt, "parallel", None) == "DDP"
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_rank() > 0
        )

        if self._is_ddp_worker:
            # Hard disable W&B on workers to avoid any API and background threads
            os.environ["WANDB_MODE"] = "disabled"
            os.environ["WANDB_SILENT"] = "true"
            os.environ["WANDB_DISABLE_CODE"] = "true"
            self._enabled = False
            return

        # Main rank only from here
        if os.environ.get("WANDB_MODE") != "offline":
            api_key = os.environ.get("WANDB_API_KEY", None)
            wandb.login(key=api_key)

        # WandB init
        wandb.init(
            project="DeepEM",
            name=opt.exp_name,
            resume="allow",
            id=opt.exp_name,
            # settings=wandb.Settings(
            #     start_method="thread",
            #     _disable_stats=True  # optional: reduce background metrics chatter
            # ),
        )
        wandb.config.update(opt, allow_val_change=True)
        self._enabled = True

    def __enter__(self) -> None:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> None:
        if getattr(self, "_enabled", False):
            wandb.finish()

    def log_metrics(self,
        phase: str,
        iter_num: int,
        stats: dict[str, float],
    ) -> None:
        if not getattr(self, "_enabled", False):
            return
        wandb.log(
            {f"{phase}/{metric}": value for metric, value in stats.items()},
            step=iter_num
        )

    def log_images(self,
        phase: str,
        iter_num: int,
        preds: dict[str, torch.Tensor],
        sample: dict[str, torch.Tensor],
    ) -> None:
        """Log 3D images."""
        if not getattr(self, "_enabled", False):
            return
        # Get reference sizes
        out_key = sorted(self.out_spec)[0]
        in_key = sorted(self.in_spec)[0]
        outsz = sample[out_key].shape[-3:]
        insz = sample[in_key].shape[-3:]

        # Input
        logs = []
        for key in sorted(self.in_spec):
            # Crop inputs to match output size if not padding
            cropsz = outsz if not self.pad_output else None
            logs.append(wandb.Image(self.to_array(sample[key], cropsz=cropsz), caption=key))

        # Outputs
        for key in sorted(self.out_spec):
            # Pad outputs to match input size if padding
            padsz = insz if self.pad_output else None

            # Prediction
            if key in ["embedding"]:
                # Metric graph
                aff = vec2aff(preds[key], delta_d=self.opt.delta_d)
                arr = self.to_array(aff, padsz=padsz)
                logs.append(wandb.Image(arr, caption=f"{key} metric graph"))

                # Embeddings
                vec = preds[key][[0],...].cpu()
                vec = torch_utils.vec2pca(vec)
                arr = self.to_array(vec.select(0, 0), padsz=padsz)
            elif key in ["mitochondria_embedding"]:
                # Mitochondria embedding
                vec = preds[key][[0],...].cpu()
                vec = torch_utils.vec2pca(vec)
                arr = self.to_array(vec.select(0, 0), padsz=padsz)
            else:
                arr = self.to_array(torch.sigmoid(preds[key]), padsz=padsz)
            logs.append(wandb.Image(arr, caption=f"{key} prediciton"))

            # Label
            if key in ["affinity", "long_range", "embedding", "mitochondria_embedding"]:
                seg = sample[key][0,0,...].cpu().numpy().astype('uint32')
                rgb = torch.from_numpy(py_utils.seg2rgb(seg))
                arr = self.to_array(rgb, padsz=padsz)
            else:
                arr = self.to_array(sample[key], padsz=padsz)
            logs.append(wandb.Image(arr, caption=f"{key} label"))

            # Mask
            arr = self.to_array(sample[f"{key}_mask"], padsz=padsz)
            logs.append(wandb.Image(arr, caption=f"{key} mask"))

        # Log images
        wandb.log({f"results/{phase}_slider": logs}, step=iter_num)

    def to_array(
        self,
        tensor: torch.Tensor,
        cropsz: tuple[int, int, int] | None = None,
        padsz: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        """Convert a tensor to a loggable array."""
        tensor = tensor.cpu()
        if padsz is not None:
            # Pad smaller tensors to match padsz
            if any(t < p for t, p in zip(tensor.shape[-3:], padsz)):
                tensor = torch_utils.pad_center(tensor, padsz)
        if cropsz is not None:
            # Crop larger tensors to match cropsz
            tensor = torch_utils.crop_center_no_strict(tensor, cropsz)
        assert (tensor.ndim >= 3) and (tensor.ndim <= 5)
        tensor = tensor[0, ...] if tensor.ndim > 4 else tensor
        tensor = tensor[0:3, ...] if tensor.ndim > 3 else tensor
        depth = tensor.shape[-3]
        imgs = [tensor[:,z,:,:] for z in range(depth)]
        return make_grid(imgs, nrow=depth, padding=0)
