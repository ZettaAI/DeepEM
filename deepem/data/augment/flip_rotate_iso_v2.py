from augmentor import *


def get_augmentation(is_train, recompute=False, border=False, **kwargs):
    augs = list()

    # Recompute connected components
    if recompute:
        augs.append(Label())

    # Create border
    if border:
        augs.append(Border())

    # Flip & rotate
    augs.append(FlipRotateIsotropic())

    return Compose(augs)
