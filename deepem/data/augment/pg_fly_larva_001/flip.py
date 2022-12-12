"""Flip/rotate Isotropic
"""
from augmentor import *


def get_augmentation(
    is_train, **kwargs
):
    augs = list()

    # Flip & rotate
    augs.append(FlipRotateIsotropic())

    return Compose(augs)
