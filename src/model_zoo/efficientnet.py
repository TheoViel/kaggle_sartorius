import timm
import numpy as np
import torch.nn as nn
from mmdet.models.builder import BACKBONES

from params import MEAN, STD


@BACKBONES.register_module()
class EfficientNet(nn.Module):
    def __init__(self, name, blocks_idx, pretrained=True):
        super().__init__()

        self.effnet = getattr(timm.models, name)(
            pretrained=pretrained,
            drop_path_rate=0.2,
        )

        self.block_idx = blocks_idx
        self.nb_fts = [self.effnet.blocks[b][-1].conv_pwl.out_channels for b in self.block_idx]
        self.nb_ft = self.nb_fts[-1]

        if "efficientnetv2" in name:
            self.mean = np.array([0.5, 0.5, 0.5])
            self.std = np.array([0.5, 0.5, 0.5])
        else:
            self.mean = MEAN
            self.std = STD

        self.name = name

    def forward(self, x):  # should return a tuple
        """
        Extract features for an EfficientNet model.
        Args:
            x (torch tensor [BS x 3 x H x W]): Input image.
        Returns:
            torch tensors: features.
        """
        x = self.effnet.conv_stem(x)
        x = self.effnet.bn1(x)
        x = self.effnet.act1(x)

        features = []
        for i, b in enumerate(self.effnet.blocks):
            x = b(x)
            if i in self.block_idx:
                features.append(x)
            print(i, x.size(), i in self.block_idx)

        return features
