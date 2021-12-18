import torch.nn as nn

from model_zoo.encoders import get_encoder


def define_model(name, num_classes=3, num_classes_aux=9):
    """
    Loads the model.
    Args:
        name (str): Name of the encoder.
        num_classes (int, optional): Number of classes to use. Defaults to 1.
    Returns:
        torch model: Model.
    """
    encoder = get_encoder(name)
    model = SartoriusModel(encoder, num_classes=num_classes, num_classes_aux=num_classes_aux)
    return model


class SartoriusModel(nn.Module):
    def __init__(self, encoder, num_classes=1, num_classes_aux=1):
        """
        Constructor.
        Args:
            encoder (nn.Module): encoder to build the model from.
            num_classes (int, optional): Number of classes to use. Defaults to 1.
        """
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.nb_ft = encoder.nb_ft
        self.mean = encoder.mean
        self.std = encoder.std

        self.logits = nn.Linear(self.nb_ft, num_classes)
        self.logits_aux = nn.Linear(self.nb_ft, num_classes_aux)

    def forward(self, x):
        """
        Usual torch forward function
        Args:
            x (torch tensor [BS x 3 x H x W]): Input image
        Returns:
            torch tensor [BS x NUM_CLASSES]: Study logits.
            torch tensor [BS x 1]: Image logits.
            list or torch tensors : Masks.
        """
        x1, x2, x3, x4 = self.encoder.extract_features(x)

        pooled = x4.mean(-1).mean(-1)

        logits = self.logits(pooled)
        logits_aux = self.logits_aux(pooled)

        return logits, logits_aux
