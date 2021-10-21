import numpy as np
import scipy.ndimage as ndi


def rles_to_mask(encs, shape):
    """
    Decodes a rle.

    Args:
        encs (list of str): Rles for each class.
        shape (tuple [2]): Mask size.

    Returns:
        np array [shape]: Mask.
    """
    img = np.zeros(shape[0] * shape[1], dtype=np.uint)
    for m, enc in enumerate(encs):
        if isinstance(enc, np.float) and np.isnan(enc):
            continue
        enc_split = enc.split()
        for i in range(len(enc_split) // 2):
            start = int(enc_split[2 * i]) - 1
            length = int(enc_split[2 * i + 1])
            img[start: start + length] = 1 + m
    return img.reshape(shape)


def rles_to_mask_fix(encs, shape, single_channel=True):
    """
    Decodes a rle.

    Args:
        encs (list of str): Rles for each class.
        shape (tuple [2]): Mask size.

    Returns:
        np array [shape]: Mask.
    """
    img = np.zeros((len(encs), shape[0] * shape[1]), dtype=np.uint16)
    for m, enc in enumerate(encs):
        if isinstance(enc, np.float) and np.isnan(enc):
            continue
        enc_split = enc.split()
        for i in range(len(enc_split) // 2):
            start = int(enc_split[2 * i]) - 1
            length = int(enc_split[2 * i + 1])
            img[m, start: start + length] = 1
    img = img.reshape(len(encs), shape[0], shape[1])

    mask = np.zeros(img.shape)
    for i in range(len(img)):
        mask[i] = ndi.morphology.binary_fill_holes(img[i]).astype(int) * (i + 1)

    if single_channel:
        return mask.max(0)
    else:
        return mask


def mask_to_rles(mask):
    pixels = mask.flatten()

    # This simplified method requires first and last pixel to be zero
    pixels[0] = 0
    pixels[-1] = 0

    rles = []
    for i in range(1, int(np.max(pixels)) + 1):
        pix = (pixels == i)
        runs = np.where(pix[1:] != pix[:-1])[0] + 2
        runs[1::2] -= runs[::2]
        rles.append(" ".join(str(x) for x in runs))

    return rles
