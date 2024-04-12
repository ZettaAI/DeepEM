"""Based on flyem/aug_mip1.py with some modifications.

Mostly removing things that we might not need for faster
feedback cycles.
"""
from augmentor import *


def get_augmentation(
    is_train,
    recompute=False,
    box=None,
    blur=7,
    border=[],
    **kwargs
):
    augs = list()

    # Flip & rotate
    augs.append(FlipRotate())

    # Brightness & contrast perturbation
    augs.append(
        Grayscale3D(
            contrast_factor=0.5,
            brightness_factor=0.5,
            skip=0.3,
        )
    )

    # Box
    if is_train:
        if box == "fill":
            augs.append(
                Blend([
                    FillBox(
                        random=True,
                        dims=(5, 25),
                        aniso=1,
                        density=0.3,
                        margin=(5, 5, 5),
                        individual=True,
                        skip=0.1,
                    ),
                    FillBox(
                        random=True,
                        dims=(5, 25),
                        aniso=1,
                        density=0.3,
                        margin=(5, 5, 5),
                        individual=False,
                        skip=0.1,
                    ),
                ])
            )

    # Out-of-focus section
    if blur > 0:
        augs.append(MixedBlurrySection(maxsec=blur))

    # Warping
    if is_train:
        augs.append(Warp(skip=0.3, do_twist=False, rot_max=45.0, scale_max=1.1))

    # Flip & rotate
    augs.append(FlipRotate())

    # Create border
    if border:
        augs.append(Border(targets=border))

    # Recompute connected components
    if recompute:
        augs.append(Label())

    return Compose(augs)
