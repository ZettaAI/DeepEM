import os
import sys
import time

import numpy as np
import torch
import samwise
import napari

from deepem.train.option import Options
from deepem.train.utils import *


def sample(opt):

    if opt.batch_size != 1 or opt.num_workers != 1:
        warnings.warn("Setting batch size and # workers to 1 for now")
        opt.batch_size = 1
        opt.num_workers = 1

    # Data loaders
    train_loader, val_loader = load_data(opt)

    # Sample loop
    t0 = time.time()

    viewer = napari.Viewer()

    @viewer.bind_key("q")
    def quit(viewer):
        sys.exit()

    def convert(t: torch.Tensor) -> np.ndarray:
        return t.cpu().detach().numpy().astype(np.uint8)

    print("Press ENTER to continue, Q to quit")
    for i in range(opt.chkpt_num, opt.max_iter):

        with torch.no_grad():
            # Load training samples.
            sample = train_loader()

            # Elapsed time
            elapsed = time.time() - t0
            print(f"Sample generated in {elapsed:.3f}s") 

            # Viewer
            viewer.add_image(convert(sample["input"] * 255), name="image")
            for k in opt.out_spec:
                viewer.add_labels(convert(sample[k]), name=k, opacity=0.5)
                viewer.add_image(
                    convert(sample[f"{k}_mask"] * 255), name=f"{k}_mask", opacity=0.2
                )

            # Wait for user input (enter to continue, q to quit)
            input()

            # Reset timer and viewer.
            for layer in range(len(viewer.layers)):
                viewer.layers.pop()
            t0 = time.time()


if __name__ == "__main__":
    # Options
    opt = Options().parse()
    sample(opt)
