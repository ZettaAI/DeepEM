from augmentor import *
from augmentor.degradation import ImageDegradation, ImageDegradation2D


def get_augmentation(
    is_train,
    box=None,
    missing=7,
    blur=7,
    lost=True,
    random=False,
    recompute=[],
    border=[],
    section_gap=0,
    mask_section_gap=False,
    degradation_skip=0.1,
    **kwargs
):
    """Augmentation spec with harsh SNR degradation for robust training.

    This spec adds a blended ImageDegradation (50% 2D + 50% 3D) that spans
    from mild to nightmare levels, simulating severe image quality issues
    (shot noise, readout noise, poor contrast).

    Args:
        degradation_skip: Skip probability for each degradation component
            (Poisson, Gaussian, contrast). Default 0.1. With skip=0.1, ~73%
            of samples get full degradation, ~27% partial, ~0.1% minimal.
            With skip=0.2, ~51% full, ~48% partial, ~0.8% minimal.

    Degradation Parameters (nightmare-spanning):
        peak_electrons: (5, 200)      # Poisson noise: nightmare(5) → mild(200)
        variance:       (0.005, 0.06) # Gaussian noise: moderate → nightmare+
        compression:    (0.1, 0.6)    # Contrast: nightmare(0.1) → moderate(0.6)

    Blended approach:
        - 2D (per-slice): Simulates slice-to-slice variability in EM
        - 3D (volume-wide): Simulates systematic degradation across a block

    The combination of all three at extreme values creates nightmare-level
    degradation. With skip probabilities, some samples will be cleaner.
    """
    augs = list()

    # Box
    if is_train:
        if box == 'noise':
            augs.append(
                NoiseBox(sigma=(1, 3), dims=(10, 50), margin=(1, 10, 10),
                         density=0.3, skip=0.1)
            )
        elif box == 'fill':
            augs.append(
                FillBox(dims=(10, 50), margin=(1, 10, 10),
                        density=0.3, skip=0.1)
            )

    # ==========================================================================
    # Image Degradation (Harsh SNR degradation - nightmare spanning)
    # ==========================================================================
    # Physical model (Sardhara et al., 2022):
    #   1. Poisson noise → Shot noise from electron counting
    #   2. Gaussian noise → Electronic readout interference
    #   3. Contrast compression → Poor staining / thick sections
    #
    # Parameter ranges span from mild to nightmare:
    #   - peak_electrons (5, 200): Lower = more Poisson noise
    #   - variance (0.005, 0.06): Higher = more Gaussian noise
    #   - compression_factor (0.1, 0.6): Lower = less contrast
    #
    # Blended approach (50% 2D + 50% 3D):
    #   - 2D: Per-slice variability (realistic for EM section-to-section noise)
    #   - 3D: Volume-wide consistency (systematic degradation across block)
    #
    # With default skip=0.1, ~73% of samples get full degradation pipeline,
    # ~27% get partial degradation, ~0.1% get minimal/no degradation.
    # With skip=0.2, ~51% full, ~48% partial, ~0.8% minimal/no degradation.
    # ==========================================================================
    if is_train:
        degradation_params = dict(
            peak_electrons=(5, 200),
            variance=(0.005, 0.06),
            compression_factor=(0.1, 0.6),
            mean_shift=(-0.1, 0.1),
            use_local_mean=True,
            skip_poisson=degradation_skip,
            skip_gaussian=degradation_skip,
            skip_contrast=degradation_skip,
        )
        augs.append(
            Blend([
                ImageDegradation2D(prob=1.0, **degradation_params),  # Per-slice
                ImageDegradation(**degradation_params),              # Volume-wide
            ])
        )

    # Brightness & contrast perturbation
    augs.append(
        MixedGrayscale2D(
            contrast_factor=0.5,
            brightness_factor=0.5,
            prob=1, skip=0.3))

    # Missing section & misalignment
    to_blend = list()
    # Misalingments
    trans = Compose([Misalign((0, 10), margin=1),
                     Misalign((0, 30), margin=1),
                     Misalign((0, 50), margin=1)])

    # Out-of-alignments
    slip = Compose([SlipMisalign((0, 10), interp=True, margin=1),
                    SlipMisalign((0, 30), interp=True, margin=1),
                    SlipMisalign((0, 50), interp=True, margin=1)])
    to_blend.append(Blend([trans, slip], props=[0.7, 0.3]))
    if is_train:
        to_blend.append(Blend([
            MisalignPlusMissing((5, 30), value=0, random=random),
            MisalignPlusMissing((5, 30), value=0, random=False)
        ]))
    else:
        to_blend.append(MisalignPlusMissing((5, 30), value=0, random=False))
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
    if lost:
        if is_train:
            to_blend.append(Blend([
                LostSection(1),
                LostPlusMissing(value=0, random=random),
                LostPlusMissing(value=0, random=False)
            ]))
    augs.append(Blend(to_blend))

    # Out-of-focus
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
        augs.append(Label(targets=recompute))

    # Section gap
    if section_gap > 0:
        augs.append(SectionGap(num_secs=section_gap, masked=mask_section_gap))

    return Compose(augs)
