import numpy as np
from scipy.ndimage.morphology import binary_fill_holes


def rle_decode(mask_rle, shape):
    """
    Decodes a rle.

    Args:
        mask_rle (str): Run length encoding.
        shape (tuple [2]): Mask size (height, width).

    Returns:
        np array [shape]: Mask.
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo: hi] = 1
    return img.reshape(shape)


def rles_to_mask_fix(encs, shape, single_channel=True, fix=True):
    """
    Decodes rles and optionally fixes broken masks with fill_poly.

    Args:
        encs (list of str): Rles for each class.
        shape (tuple [2]): Mask size (height, width).
        single_channel (bool, Optional): Whether to convert the mask to single channel.
        fix (bool, Optional): Whether to dix masks.

    Returns:
        np array [shape]: Mask.
    """
    masks = np.zeros((len(encs), shape[0], shape[1]), dtype=np.uint16)
    for m, enc in enumerate(encs):
        masks[m] = rle_decode(enc, shape)

    masks_fixed = np.zeros((len(encs), shape[0], shape[1]), dtype=np.uint16)
    if fix:
        for i in range(len(masks)):
            masks_fixed[i] = binary_fill_holes(masks[i]).astype(int) * (i + 1)
    else:
        for i in range(len(masks)):
            masks_fixed[i] = masks[i].astype(int) * (i + 1)

    if single_channel:
        return masks_fixed.max(0)
    else:
        return masks_fixed


def rle_encode(img):
    """
    TODO
    Not used except for inference.

    Args:
        img ([type]): [description]

    Returns:
        [type]: [description]
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
