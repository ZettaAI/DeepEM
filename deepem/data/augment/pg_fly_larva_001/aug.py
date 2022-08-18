"""Based on flyem/aug_mip1.py with some modifications.

Mostly removing things that we might not need for faster
feedback cycles.
"""
from augmentor import *


def get_augmentation(
    is_train, box=None, missing=7, blur=7, lost=True, random=False, **kwargs
):
    augs = list()

    # Flip & rotate
    augs.append(FlipRotateIsotropic())

    # Brightness & contrast perturbation
    augs.append(
        MixedGrayscale2D(
            contrast_factor=0.5,
            brightness_factor=0.5,
            prob=1,
            skip=0.7,
        )
    )

    # Box
    if is_train:
        if box == "noise":
            augs.append(
                NoiseBox(
                    sigma=(1, 3),
                    dims=(5, 25),
                    margin=(1, 5, 5),
                    density=0.3,
                    skip=0.8,
                    aniso=1,
                )
            )
        elif box == "fill":
            augs.append(
                FillBox(
                    dims=(5, 25),
                    margin=(5, 5, 5),
                    density=0.1,
                    skip=1,
                    aniso=1,
                )
            )

    # Out-of-focus section
    if blur > 0:
        augs.append(MixedBlurrySection(maxsec=blur))

    # Warping
    if is_train:
        augs.append(Warp(skip=0.3, do_twist=False, rot_max=45.0, scale_max=1.1))

    # Flip & rotate
    augs.append(FlipRotateIsotropic())

    return Compose(augs)
