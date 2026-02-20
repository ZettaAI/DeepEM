from augmentor import *


def get_augmentation(
    is_train,
    box=None,
    missing=7,
    blur=7,
    lost=True,
    random=False,
    recompute=[],
    border=[],
    **kwargs
):
    augs = list()
    if not is_train:
        return Compose(augs)

    # Brightness & contrast purterbation
    augs.append(
        MixedGrayscale2D(
            contrast_factor=0.5,
            brightness_factor=0.5,
            prob=1, skip=0.3))

    # Mutually exclusive augmentations
    mutex = list()

    # (1) Misalingment
    trans = Compose([Misalign((0, 5), margin=1),
                     Misalign((0,15), margin=1),
                     Misalign((0,25), margin=1)])
    slip = Compose([SlipMisalign((0, 5), interp=True, margin=1),
                    SlipMisalign((0,15), interp=True, margin=1),
                    SlipMisalign((0,25), interp=True, margin=1)])
    mutex.append(Blend([trans,slip], props=[0.7,0.3]))

    # (2) Misalignment + missing section
    mutex.append(Blend([
        MisalignPlusMissing((3,15), value=0, random=random),
        MisalignPlusMissing((3,15), value=0, random=False)
    ]))

    # (3) Missing section
    if missing > 0:
        mutex.append(Blend([
            MixedMissingSection(maxsec=missing, individual=True, value=0, random=False),
            MixedMissingSection(maxsec=missing, individual=True, value=0, random=random),
            MissingSection(maxsec=missing, individual=False, value=0, random=random),
        ]))

    # (4) Lost section
    if lost:
        mutex.append(Blend([
            LostSection(1),
            LostPlusMissing(value=0, random=random),
            LostPlusMissing(value=0, random=False)
        ]))

    # Mutually exclusive augmentations
    augs.append(Blend(mutex))

    # Box
    if box == 'noise':
        augs.append(
            NoiseBox(sigma=(1,3), dims=(5,25), margin=(1,5,5),
                        density=0.3, skip=0.1)
        )
    elif box == 'fill':
        augs.append(
            FillBox(dims=(5,25), margin=(1,5,5),
                    density=0.3, skip=0.1)
        )

    # Out-of-focus section
    if blur > 0:
        augs.append(MixedBlurrySection(maxsec=blur))

    # Warping
    augs.append(Warp(skip=0.3, do_twist=False, rot_max=45.0, scale_max=1.1))

    # Simplex warp field (non-rigid deformation)
    augs.append(
        SimplexNoiseWarpField(
            skip=0.01, max_displacement=16.0, frequency=0.005, octaves=5, z_anisotropy=None
        )
    )

    # Recompute connected components
    if recompute:
        augs.append(Label(targets=recompute))

    # Flip & rotate
    augs.append(FlipRotate())

    # Create border
    if border:
        augs.append(Border(targets=border))

    return Compose(augs)
