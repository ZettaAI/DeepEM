from augmentor import *


def get_augmentation(
    is_train,
    box=None,
    missing=7,
    blur=7,
    lost=True,
    random=False,
    recompute=False,
    border=[],
    **kwargs
):
    augs = list()

    # Box
    if is_train:
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

    # Brightness & contrast purterbation
    augs.append(
        MixedGrayscale2D(
            contrast_factor=0.5,
            brightness_factor=0.5,
            prob=1, skip=0.3))

    # Mutually-exclusive data augmentations
    to_blend = list()
    
    # Step
    step = Compose([Misalign((0, 5), margin=1),
                    Misalign((0,15), margin=1),
                    Misalign((0,25), margin=1)])

    # Slip
    slip = Compose([SlipMisalign((0, 5), interp=True, margin=1),
                    SlipMisalign((0,15), interp=True, margin=1),
                    SlipMisalign((0,25), interp=True, margin=1)])
    
    # Step + slip
    to_blend.append(Blend([step, slip], props=[0.7, 0.3]))

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
    if lost:
        if is_train:
            to_blend.append(
                Blend(
                    [
                        LostSection(1),
                        LostSection(2),
                        LostSection(3),
                        LostPlusMissing(value=0, random=random),
                        LostPlusMissing(value=0, random=False),
                    ],
                    props=[0.4, 0.3, 0.2, 0.05, 0.05],
                )
            )
    augs.append(Blend(to_blend))

    # Out-of-focus sections
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
