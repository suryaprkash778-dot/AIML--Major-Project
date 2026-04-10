# dataset.py
import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import config


def get_cifar10_dataloaders():
    """
    Downloads CIFAR-10 and returns standard PyTorch DataLoaders.
    Note: MixUp/CutMix are applied at the *batch* level in the training loop,
    so they are not included in these baseline transforms.
    """

    # Modern PyTorch v2 transforms (faster, tensor-native)
    train_transform = v2.Compose([
        v2.ToImage(),  # Converts to tensor
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        # Standard mathematical normalization values specifically for CIFAR-10
        v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

    # Test data should never be augmented, only converted and normalized
    test_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

    # Download and load the datasets into a folder called 'data'
    print("Loading CIFAR-10 Dataset...")
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    # Create DataLoaders to feed the neural network in batches
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True  # Speeds up transfer to GPU
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, test_loader


# Quick test block: if you run this file directly, it checks if your code works
if __name__ == "__main__":
    train_dl, test_dl = get_cifar10_dataloaders()
    images, labels = next(iter(train_dl))
    print(f"Success! Batch of images shape: {images.shape}")
    print(f"Success! Batch of labels shape: {labels.shape}")
