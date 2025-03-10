from __future__ import annotations

import argparse
import os
from types import TracebackType

import numpy as np
import torch
from torchvision.utils import make_grid
import wandb

from deepem.loss.mean import vec2aff
from deepem.utils import py_utils, torch_utils


class WandbLogger:
    """
    Weight & Biases Logger.
    """

    def __init__(
        self,
        opt: argparse.Namespace,
    ):
        self.opt = opt
        self.in_spec = dict(opt.in_spec)
        self.out_spec = dict(opt.out_spec)

        # WandB login
        if not os.environ.get("WANDB_MODE", None) == "offline":  # pragma: no cover
            api_key = os.environ.get("WANDB_API_KEY", None)
            wandb.login(key=api_key)

        # WandB init
        wandb.init(
            project="DeepEM",
            name=opt.exp_name,
            resume="allow",
            id=opt.exp_name,
        )
        wandb.config.update(opt, allow_val_change=True)

    def __enter__(self) -> None:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> None:
        wandb.finish()

    def log_metrics(self,
        phase: str,
        iter_num: int,
        stats: dict[str, float],
    ) -> None:
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
        # Peep output size
        key = sorted(self.out_spec)[0]
        cropsz = sample[key].shape[-3:]
        for k in sorted(self.out_spec):
            outsz = sample[k].shape[-3:]
            assert np.array_equal(outsz, cropsz)

        # Input
        logs = []
        for key in sorted(self.in_spec):
            logs.append(wandb.Image(self.to_array(sample[key], cropsz), caption=key))

        # Outputs
        for key in sorted(self.out_spec):

            # Prediction
            if key in ["embedding"]:
                # Metric graph
                aff = vec2aff(preds[key], delta_d=self.opt.delta_d)
                arr = self.to_array(aff)
                logs.append(wandb.Image(arr, caption=f"{key} metric graph"))

                # Embeddings
                vec = preds[key][[0],...].cpu()
                vec = torch_utils.vec2pca(vec)
                arr = self.to_array(vec.select(0, 0))
            else:
                arr = self.to_array(torch.sigmoid(preds[key]))
            logs.append(wandb.Image(arr, caption=f"{key} prediciton"))

            # Label
            if key in ["affinity", "long_range", "embedding"]:
                seg = sample[key][0,0,...].cpu().numpy().astype('uint32')
                rgb = torch.from_numpy(py_utils.seg2rgb(seg))
                arr = self.to_array(rgb)
            else:
                arr = self.to_array(sample[key])
            logs.append(wandb.Image(arr, caption=f"{key} label"))

            # Mask
            arr = self.to_array(sample[f"{key}_mask"])
            logs.append(wandb.Image(arr, caption=f"{key} mask"))

        # Log images
        wandb.log({f"results/{phase}_slider": logs}, step=iter_num)

    def to_array(
        self,
        tensor: torch.Tensor,
        cropsz: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        """Convert a tensor to a loggable array."""
        tensor = tensor.cpu()
        if cropsz is not None:
            tensor = torch_utils.crop_center_no_strict(tensor, cropsz)
        assert (tensor.ndim >= 3) and (tensor.ndim <= 5)
        tensor = tensor[0, ...] if tensor.ndim > 4 else tensor
        tensor = tensor[0:3, ...] if tensor.ndim > 3 else tensor
        depth = tensor.shape[-3]
        imgs = [tensor[:,z,:,:] for z in range(depth)]
        return make_grid(imgs, nrow=depth, padding=0)
