import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import albumentations as albu  # noqa
from albumentations import pytorch as AT # noqa

from params import MEAN, STD, SIZE # noqa


def noise_transforms(p=0.5):
    """
    Applies GaussNoise or RandomFog random with a probability p.
    Args:
        p (float, optional): probability. Defaults to 0.5.
    Returns:
        albumentation transforms: transforms.
    """
    return albu.OneOf(
        [
            albu.GaussNoise(var_limit=(1.0, 50.0), always_apply=True),
            albu.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.25, always_apply=True),
        ],
        p=p,
    )


def blur_transforms(p=0.5):
    """
    Applies MotionBlur, GaussianBlur or RandomFog random with a probability p.
    Args:
        p (float, optional): probability. Defaults to 0.5.
    Returns:
        albumentation transforms: transforms.
    """
    return albu.OneOf(
        [
            albu.MotionBlur(blur_limit=5, always_apply=True),
            albu.GaussianBlur(blur_limit=9, always_apply=True),
            albu.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.2, always_apply=True),
        ],
        p=p,
    )


def color_transforms(p=0.5):
    """
    Applies RandomGamma or RandomBrightnessContrast random with a probability p.
    Args:
        p (float, optional): probability. Defaults to 0.5.
    Returns:
        albumentation transforms: transforms.
    """
    return albu.OneOf(
        [
            albu.RandomGamma(gamma_limit=(70, 130), always_apply=True),
            albu.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
        ],
        p=p,
    )


def distortion_transforms(p=0.5):
    """
    Applies ElasticTransform or IAAPiecewiseAffine with a probability p.
    Args:
        p (float, optional): probability. Defaults to 0.5.
    Returns:
        albumentation transforms: transforms.
    """

    return albu.OneOf(
        [
            albu.ElasticTransform(
                alpha=10,
                sigma=5.,
                alpha_affine=5.,
            ),
            albu.IAAPiecewiseAffine(
                scale=(0.03, 0.03),
                nb_rows=6,
                nb_cols=6,
            ),
        ],
        p=p,
    )


def get_transfos(size=SIZE, augment=True, visualize=False, mean=MEAN, std=STD):
    """
    Returns transformations for the OCT images.
    This version ensures masks keep a meaningful shape.
    Args:
        size (int, optional): Image will be resized to (size, size * ratio). Defaults to SIZE.
        augment (bool, optional): Whether to apply augmentations. Defaults to True.
        visualize (bool, optional): Whether to use transforms for visualization. Defaults to False.
        mean (np array [3], optional): Mean for normalization. Defaults to MEAN.
        std (np array [3], optional): Standard deviation for normalization. Defaults to STD.
    Returns:
        albumentation transforms: transforms.
    """
    if visualize:
        normalizer = albu.Compose(
            [
                AT.transforms.ToTensorV2(),
            ],
            p=1,
        )
    else:
        normalizer = albu.Compose(
            [
                albu.Normalize(mean=mean, std=std),
                AT.transforms.ToTensorV2(),
            ],
            p=1,
        )

    if augment:
        return albu.Compose(
            [
                albu.RandomCrop(size, size),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                # noise_transforms(p=0.5),
                # color_transforms(p=0.5),
                # blur_transforms(p=0.5),
                # albu.CLAHE(p=0.1),
                normalizer,
            ]
        )
    else:
        return albu.Compose(
            [
                normalizer,
            ]
        )


def get_transfos_inference(mean=MEAN, std=STD):
    """
    Returns inference augmentations.
    Args:
        mean (np array [3], optional): Mean for normalization. Defaults to MEAN.
        std (np array [3], optional): Standard deviation for normalization. Defaults to STD.
    Returns:
        albumentation transforms: transforms.
    """
    return albu.Compose(
        [
            albu.PadIfNeeded(
                min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32
            ),
            albu.Normalize(mean=mean, std=std),
            AT.transforms.ToTensorV2(),
        ],
    )
