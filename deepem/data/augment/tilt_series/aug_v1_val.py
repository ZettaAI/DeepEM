from augmentor import *


def get_augmentation(is_train, tilt_series=(0,0,0), tilt_series_crop=None,
                     recompute=False, box=None, missing=7, blur=7, random=False,
                     **kwargs):
    augs = []

    # Flip & rotate (isotropic)
    if not is_train:
        augs.append(FlipRotateIsotropic())

    # Tilt series projection & label subsampling
    if tilt_series[0] > 0:
        ts_n = tilt_series[0]
        ts_in = tilt_series[1]
        ts_out = tilt_series[2]
        assert ts_in > 0 and ts_out > 0
        assert ts_in % ts_out == 0
        if ts_n == 1: 
            augs.append(TiltSeries(ts_in, projections=[NormalView(ts_in)]))
        else:
            assert ts_n == 5
            augs.append(TiltSeries(ts_in))
        augs.append(SubsampleLabels(factor=(ts_out,1,1)))
        if tilt_series_crop is not None:
            augs.append(CropLabels(tilt_series_crop))

    # Recompute connected components
    if recompute:
        augs.append(Label())

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
    if is_train:
        augs.append(
            MixedGrayscale2D(
                contrast_factor=0.5,
                brightness_factor=0.5,
                prob=1, skip=0.3))

    # Missing section & misalignment
    to_blend = list()

    # Misalingments
    trans = Compose([Misalign((0, 5), margin=1),
                        Misalign((0,15), margin=1),
                        Misalign((0,25), margin=1)])

    # Out-of-alignments
    slip = Compose([SlipMisalign((0, 5), interp=True, margin=1),
                    SlipMisalign((0,15), interp=True, margin=1),
                    SlipMisalign((0,25), interp=True, margin=1)])
    
    to_blend.append(Blend([trans,slip], props=[0.7,0.3]))

    # Misalign + missing
    to_blend.append(Blend([
        MisalignPlusMissing((3,15), value=0, random=random),
        MisalignPlusMissing((3,15), value=0, random=False)
    ]))

    # Missing section
    if missing > 0:
        to_blend.append(Blend([
            MixedMissingSection(maxsec=missing, individual=True, value=0, random=False),
            MixedMissingSection(maxsec=missing, individual=True, value=0, random=random),
            MissingSection(maxsec=missing, individual=False, value=0, random=random),
        ]))
    
    if is_train:
        augs.append(Blend(to_blend))
        
    # Out-of-focus
    if is_train and blur > 0:
        augs.append(MixedBlurrySection(maxsec=blur))

    # Warping
    if is_train:
        augs.append(Warp(skip=0.3, do_twist=False, rot_max=45.0, scale_max=1.1))
    
    # Flip & rotate
    if is_train:
        augs.append(FlipRotate())

    return Compose(augs)