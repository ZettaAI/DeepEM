from augmentor import *


def get_augmentation(is_train, recompute=[], **kwargs):
    augs = list()

    # Recompute connected components
    if recompute:
        augs.append(Label(targets=recompute))

    # Flip & rotate
    augs.append(FlipRotate())

    return Compose(augs)
