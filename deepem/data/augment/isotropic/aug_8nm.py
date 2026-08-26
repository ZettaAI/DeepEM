"""
Isotropic augmentation for 8 nm data fed to an up-down model.

Copy of the isotropic branch of super_resolution/aug_v0.py -- exactly what the
260805 aff-head checkpoints were trained with -- with the occlusion boxes
doubled so they keep their physical size when the voxel size halves:

    dims   (5, 25)   ->  (10, 50)     40-200 nm  ->  80-400 nm
    margin (5, 5, 5) ->  (10, 10, 10)

`blur` is MixedBlurrySection(maxsec=...), a count of sections, so it scales the
same way -- but it is a config knob, so double it there (45 -> 90) rather than
here.

Nothing else is resolution-dependent: Grayscale3D and AdditiveGaussianNoise act
on intensity, and Warp's rot_max / scale_max are dimensionless.
"""
from augmentor import *


def get_augmentation(
    is_train,
    flip=True,
    box=None,
    blur=7,
    border=[],
    recompute=[],
    noise=None,
    **kwargs,
):
    """
    Args:
        is_train: Whether this is for training (enables more augmentations)
        flip: Enable flip/rotate augmentation
        box: Box occlusion type ('noise', 'fill', or None)
        blur: Max out-of-focus sections (0 to disable)
        border: Targets for border augmentation
        recompute: Targets for connected component recomputation
        noise: Additive Gaussian noise params (sigma_min, sigma_max, per_channel)
    """
    augs = []

    # Full 3D rotation on cubic isotropic data
    if flip:
        augs.append(FlipRotateIsotropic())

    # 3D brightness & contrast perturbation (isotropic)
    augs.append(
        Grayscale3D(
            contrast_factor=0.5,
            brightness_factor=0.5,
            skip=0.3,
        )
    )

    # Box occlusion (isotropic: aniso=1 for cubic boxes)
    if is_train:
        if box == 'noise':
            augs.append(
                NoiseBox(sigma=(1, 3), dims=(10, 50), aniso=1,
                         margin=(10, 10, 10), density=0.3, skip=0.1)
            )
        elif box == 'fill':
            augs.append(
                FillBox(dims=(10, 50), aniso=1,
                        margin=(10, 10, 10), density=0.3, skip=0.1)
            )

    # Out-of-focus sections
    if blur > 0:
        augs.append(MixedBlurrySection(maxsec=blur))

    # 3D warping
    if is_train:
        augs.append(Warp(skip=0.3, do_twist=False, rot_max=45.0, scale_max=1.1))

    # Create border (on iso-resolution labels)
    if border:
        augs.append(Border(targets=border))

    # Recompute connected components (on iso-resolution labels)
    if recompute:
        augs.append(Label(targets=recompute))

    # Additive noise
    if noise is not None:
        sigma = (noise[0], noise[1])
        per_channel = noise[2]
        augs.append(AdditiveGaussianNoise(sigma, per_channel))

    return Compose(augs)
