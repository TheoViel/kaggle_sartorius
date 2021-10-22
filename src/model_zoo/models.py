import segmentation_models_pytorch
from segmentation_models_pytorch.encoders import encoders
from params import MEAN, STD

DECODERS = [
    "Unet",
    "Linknet",
    "FPN",
    "PSPNet",
    "DeepLabV3",
    "DeepLabV3Plus",
    "PAN",
    "UnetPlusPlus",
]
ENCODERS = list(encoders.keys())


def define_model(
    decoder_name,
    encoder_name,
    num_classes=1,
    num_classes_cls=0,
    encoder_weights="imagenet",
):
    """
    Loads a segmentation architecture.

    Args:
        decoder_name (str): Decoder name.
        encoder_name (str): Encoder name.
        num_classes (int, optional): Number of classes. Defaults to 1.
        pretrained : pretrained original weights.
        encoder_weights (str, optional): Pretrained weights. Defaults to "imagenet".

    Returns:
        torch model: Segmentation model.
    """
    assert decoder_name in DECODERS, "Decoder name not supported"
    assert encoder_name in ENCODERS, "Encoder name not supported"

    decoder = getattr(segmentation_models_pytorch, decoder_name)

    model = decoder(
        encoder_name,
        encoder_weights=encoder_weights,
        classes=num_classes,
        activation=None,
        aux_params={"classes": num_classes_cls, "dropout": 0},
    )
    model.num_classes = num_classes
    model.mean = MEAN
    model.std = STD
    return model
