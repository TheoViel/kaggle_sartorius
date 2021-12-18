import timm
import torch
import numpy as np
import resnest.torch as resnest_torch

from params import MEAN, STD

BLOCKS_IDX = {
    'tf_efficientnetv2_s_in21ft1k': [1, 2, 4, 5],
    'tf_efficientnetv2_m_in21ft1k': [1, 2, 4, 6],
    'tf_efficientnetv2_l_in21ft1k': [1, 2, 4, 6],
    'tf_efficientnet_b0_ns': [1, 2, 4, 6],
    'tf_efficientnet_b1_ns': [1, 2, 4, 6],
    'tf_efficientnet_b2_ns': [1, 2, 4, 6],
    'tf_efficientnet_b3_ns': [1, 2, 4, 6],
    'tf_efficientnet_b4_ns': [1, 2, 4, 6],
    'tf_efficientnet_b5_ns': [1, 2, 4, 6],
}


def get_encoder(name):
    """
    Loads a pretrained model.
    Supports EfficientNet, ResNet, ResNext, ResNest (?) and ResNext-wsl (?).
    Args:
        name (str): Name of the model to load
        num_classes (int, optional): Number of classes to use. Defaults to 1.
    Raises:
        NotImplementedError: Unknown model name.
    Returns:
        torch model: Pretrained model
    """
    if "resnest" in name:
        model = getattr(resnest_torch, name)(pretrained=True)
    elif "wsl" in name:
        model = torch.hub.load("facebookresearch/WSL-Images", name)
    elif "resnext" in name or "resnet" in name or "densenet" in name:
        model = torch.hub.load("pytorch/vision:v0.6.0", name, pretrained=True)
    elif "efficientnet" in name:
        model = getattr(timm.models, name)(
            pretrained=True,
            drop_path_rate=0.2,
        )
    else:
        raise NotImplementedError

    if "efficientnet" in name:
        model.block_idx = BLOCKS_IDX[name]
        model.nb_fts = [model.blocks[b][-1].conv_pwl.out_channels for b in model.block_idx]
        model.nb_ft = model.nb_fts[-1]
        model.extract_features = lambda x: extract_features_efficientnet(model, x)
    else:
        model.nb_ft = model.fc.in_features
        model.nb_ft_int = model.nb_ft // 2
        model.extract_features = lambda x: extract_features_resnet(model, x)

    if "efficientnetv2" in name:
        model.mean = np.array([0.5, 0.5, 0.5])
        model.std = np.array([0.5, 0.5, 0.5])
    else:
        model.mean = MEAN
        model.std = STD

    model.name = name
    return model


def extract_features_resnet(self, x):
    """
    Extract features for a ResNet model.
    Args:
        x (torch tensor [BS x 3 x H x W]): Input image.
    Returns:
        torch tensors: features.
    """
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x1 = self.layer1(x)
    x2 = self.layer2(x1)
    x3 = self.layer3(x2)
    x4 = self.layer4(x3)

    return x1, x2, x3, x4


def extract_features_efficientnet(self, x):
    """
    Extract features for an EfficientNet model.
    Args:
        x (torch tensor [BS x 3 x H x W]): Input image.
    Returns:
        torch tensors: features.
    """
    x = self.conv_stem(x)
    x = self.bn1(x)
    x = self.act1(x)

    features = []
    for i, b in enumerate(self.blocks):
        x = b(x)
        if i in self.block_idx:
            features.append(x)

    return features
