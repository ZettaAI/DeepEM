from augmentor import *


def get_augmentation(is_train, recompute=[], border=[], **kwargs):
    augs = list()

    # Recompute connected components
    if recompute:
        augs.append(Label(targets=recompute))

    # Flip & rotate
    augs.append(FlipRotateIsotropic())

    # Create border
    if border:
        augs.append(Border(targets=border))

    return Compose(augs)
