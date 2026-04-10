# model.py
import torch
import torch.nn as nn
from torchvision.models import resnet18


def get_modified_resnet18():
    """
    Loads a standard ResNet-18 and modifies the initial layers and
    final classification head to work with 32x32 CIFAR-10 images.
    """
    model = resnet18(weights=None)

    # MODIFICATION 1: Replace the first convolutional layer
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    # MODIFICATION 2: Remove the Max Pooling layer
    model.maxpool = nn.Identity()

    # MODIFICATION 3: Change the final output layer to 10 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    return model
