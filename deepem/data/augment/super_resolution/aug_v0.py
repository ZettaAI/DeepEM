"""
Super-resolution augmentation pipeline for mixed iso/aniso training.

Architecture (completely separate pipelines):
- Isotropic data:  iso-safe augments (3D) → CubicSubsampleZ (input only)
- Anisotropic data: aniso augments (2D artifacts, misalign, missing, etc.)

Iso data gets full 3D augmentation on cubic patches. CubicSubsampleZ then
subsamples only the input in Z to simulate anisotropic acquisition, while
labels stay at isotropic resolution.

Aniso data gets standard cortex-style 2D artifact augmentation.
"""
from augmentor import *


def get_augmentation(
    is_train,
    sr_mode=False,
    sr_scale_z=5,
    flip=True,
    box=None,
    blur=7,
    border=[],
    recompute=[],
    noise=None,
    **kwargs,
):
    """
    Get augmentation pipeline for isotropic data in SR training.

    Pipeline:
    1. FlipRotateIsotropic - full 3D rotation on cubic iso data
    2. Iso-safe augmentation (3D grayscale, box, blur, warp)
    3. CubicSubsampleZ - subsample input in Z, crop labels to iso size

    Args:
        is_train: Whether this is for training (enables more augmentations)
        sr_mode: Enable super-resolution mode
        sr_scale_z: Z downsampling factor (default: 5 for 40nm:8nm ratio)
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
                NoiseBox(sigma=(1, 3), dims=(5, 25), aniso=1,
                         margin=(5, 5, 5), density=0.3, skip=0.1)
            )
        elif box == 'fill':
            augs.append(
                FillBox(dims=(5, 25), aniso=1,
                        margin=(5, 5, 5), density=0.3, skip=0.1)
            )

    # Out-of-focus sections
    if blur > 0:
        augs.append(MixedBlurrySection(maxsec=blur))

    # 3D warping
    if is_train:
        augs.append(Warp(skip=0.3, do_twist=False, rot_max=45.0, scale_max=1.1))

    # Subsample input in Z, crop labels to iso target size
    if sr_mode and sr_scale_z > 1:
        augs.append(CubicSubsampleZ(factor=sr_scale_z))

    # Create border (on iso-resolution labels)
    if border:
        augs.append(Border(targets=border))

    # Recompute connected components (on iso-resolution labels)
    if recompute:
        augs.append(Label(targets=recompute))

    # Additive noise (on aniso-resolution input)
    if noise is not None:
        sigma = (noise[0], noise[1])
        per_channel = noise[2]
        augs.append(AdditiveGaussianNoise(sigma, per_channel))

    return Compose(augs)


def get_augmentation_aniso(
    is_train,
    sr_mode=False,
    sr_scale_z=5,
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
    Get augmentation pipeline for anisotropic data in SR training.

    Standard cortex-style augmentation with 2D artifacts (misalignment,
    missing sections, etc.). No isotropic preprocessing.

    Args:
        is_train: Whether this is for training
        sr_mode: Enable super-resolution mode (unused for aniso)
        sr_scale_z: Z downsampling factor (unused for aniso)
        box: Box occlusion type ('noise', 'fill', or None)
        missing: Max missing sections (0 to disable)
        blur: Max out-of-focus sections (0 to disable)
        lost: Enable lost section augmentation
        random: Enable random missing section values
        border: Targets for border augmentation
        misalign: Misalignment magnitude (0 to disable)
        section_gap: Number of section gaps (0 to disable)
        mask_section_gap: Whether to mask section gaps
        noise: Additive Gaussian noise params (sigma_min, sigma_max, per_channel)
        recompute: Targets for connected component recomputation
    """
    augs = []

    # Box occlusion (anisotropic: aniso=sr_scale_z for flat boxes)
    if is_train:
        if box == 'noise':
            augs.append(
                NoiseBox(sigma=(1, 3), dims=(5, 25), aniso=sr_scale_z,
                         margin=(1, 5, 5), density=0.3, skip=0.1)
            )
        elif box == 'fill':
            augs.append(
                FillBox(dims=(5, 25), aniso=sr_scale_z,
                        margin=(1, 5, 5), density=0.3, skip=0.1)
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

    return Compose(augs)
