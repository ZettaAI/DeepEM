"""
Super-resolution augmentation pipeline for mixed iso/aniso training.

Architecture:
- Isotropic data: FlipRotateIsotropic (on cubic) → CubicSubsampleZ → shared aniso augmentation
- Anisotropic data: shared aniso augmentation

This allows full 3D rotations on isotropic data while sharing the core
augmentation pipeline with anisotropic data.
"""
from augmentor import *


def _get_aniso_core_augmentation(
    is_train,
    box=None,
    missing=7,
    blur=7,
    lost=True,
    random=False,
    border=[],
    misalign=0,
    section_gap=0,
    mask_section_gap=False,
    noise=None,
    recompute=[],
    **kwargs,
):
    """
    Core anisotropic augmentation pipeline shared by both iso and aniso data.

    This is the cortex-style augmentation applied after any iso-specific
    preprocessing (FlipRotateIsotropic + CubicSubsampleZ).
    """
    augs = []

    # Box occlusion (anisotropic - smaller Z margin)
    if is_train:
        if box == 'noise':
            augs.append(
                NoiseBox(sigma=(1, 3), dims=(5, 25), margin=(1, 5, 5),
                         density=0.3, skip=0.1)
            )
        elif box == 'fill':
            augs.append(
                FillBox(dims=(5, 25), margin=(1, 5, 5),
                        density=0.3, skip=0.1)
            )

    # Brightness & contrast perturbation (2D for anisotropic)
    augs.append(
        MixedGrayscale2D(
            contrast_factor=0.5,
            brightness_factor=0.5,
            prob=1, skip=0.3))

    # Missing section & misalignment (cortex-style)
    to_blend = []

    # Misalignments
    if misalign > 0 or is_train:
        trans = Compose([
            Misalign((0, 5), margin=1),
            Misalign((0, 15), margin=1),
            Misalign((0, 25), margin=1)
        ])

        # Out-of-alignments (slip)
        slip = Compose([
            SlipMisalign((0, 5), interp=True, margin=1),
            SlipMisalign((0, 15), interp=True, margin=1),
            SlipMisalign((0, 25), interp=True, margin=1)
        ])
        to_blend.append(Blend([trans, slip], props=[0.7, 0.3]))

    # Misalign plus missing
    if is_train:
        to_blend.append(Blend([
            MisalignPlusMissing((3, 15), value=0, random=random),
            MisalignPlusMissing((3, 15), value=0, random=False)
        ]))
    elif to_blend:
        to_blend.append(MisalignPlusMissing((3, 15), value=0, random=False))

    # Missing sections
    if missing > 0:
        if is_train:
            to_blend.append(Blend([
                MixedMissingSection(maxsec=missing, individual=True, value=0, random=False),
                MixedMissingSection(maxsec=missing, individual=True, value=0, random=random),
                MissingSection(maxsec=missing, individual=False, value=0, random=random),
            ]))
        else:
            to_blend.append(
                MixedMissingSection(maxsec=missing, individual=True, value=0, random=False)
            )

    # Lost sections
    if lost and is_train:
        to_blend.append(Blend([
            LostSection(1),
            LostPlusMissing(value=0, random=random),
            LostPlusMissing(value=0, random=False)
        ]))

    if to_blend:
        augs.append(Blend(to_blend))

    # Out-of-focus
    if blur > 0:
        augs.append(MixedBlurrySection(maxsec=blur))

    # Warping
    if is_train:
        augs.append(Warp(skip=0.3, do_twist=False, rot_max=45.0, scale_max=1.1))

    # Flip & rotate (XY only for anisotropic)
    augs.append(FlipRotate())

    # Create border
    if border:
        augs.append(Border(targets=border))

    # Recompute connected components
    if recompute:
        augs.append(Label(targets=recompute))

    # Section gap
    if section_gap > 0:
        augs.append(SectionGap(num_secs=section_gap, masked=mask_section_gap))

    # Additive noise
    if noise is not None:
        sigma = (noise[0], noise[1])
        per_channel = noise[2]
        augs.append(AdditiveGaussianNoise(sigma, per_channel))

    return augs


def get_augmentation(
    is_train,
    sr_mode=False,
    sr_scale_z=5,
    flip=True,
    **kwargs,
):
    """
    Get augmentation pipeline for isotropic data in SR training.

    Pipeline:
    1. FlipRotateIsotropic - full 3D rotation on cubic iso data
    2. CubicSubsampleZ - subsample Z and crop to aniso size
    3. Shared aniso augmentation (MixedGrayscale2D, misalign, etc.)

    The CubicSubsampleZ expands the spec to a cube during prepare(),
    allowing FlipRotateIsotropic to work correctly, then subsamples
    and crops back to the target aniso size.

    Args:
        is_train: Whether this is for training (enables more augmentations)
        sr_mode: Enable super-resolution mode (must be True)
        sr_scale_z: Z downsampling factor (default: 5 for 40nm:8nm ratio)
        flip: Enable flip/rotate augmentation
        **kwargs: Additional arguments passed to aniso core augmentation
    """
    augs = []

    # Full 3D rotation on cubic isotropic data
    if flip:
        augs.append(FlipRotateIsotropic())

    # Subsample Z and crop to aniso target size
    # This must come after FlipRotateIsotropic but before aniso augmentation
    if sr_mode and sr_scale_z > 1:
        augs.append(CubicSubsampleZ(factor=sr_scale_z))

    # Shared aniso core augmentation
    augs.extend(_get_aniso_core_augmentation(is_train, **kwargs))

    return Compose(augs)


def get_augmentation_aniso(
    is_train,
    sr_mode=False,
    sr_scale_z=5,
    **kwargs,
):
    """
    Get augmentation pipeline for anisotropic data in SR training.

    This simply returns the shared aniso core augmentation without
    any iso-specific preprocessing (no FlipRotateIsotropic or subsampling).

    Args:
        is_train: Whether this is for training
        sr_mode: Enable super-resolution mode (unused for aniso)
        sr_scale_z: Z downsampling factor (unused for aniso)
        **kwargs: Arguments passed to aniso core augmentation
    """
    augs = _get_aniso_core_augmentation(is_train, **kwargs)
    return Compose(augs)
