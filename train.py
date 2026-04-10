# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Importing our other files
import config
from dataset import get_cifar10_dataloaders
from model import get_modified_resnet18


# --- MIXUP MATHEMATICS ---
def mixup_data(x, y, alpha=1.0, device='cpu'):
    """Blends two images and their labels together."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Calculates the score based on the blended labels."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# -------------------------

def train():
    # 1. Setup
    print(f"Using device: {config.DEVICE}")
    train_loader, test_loader = get_cifar10_dataloaders()
    model = get_modified_resnet18().to(config.DEVICE)

    # 2. Tools for teaching (Loss function and Optimizer)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 3. The Main Training Loop
    print("Starting the training process...")
    for epoch in range(config.NUM_EPOCHS):
        model.train()  # Tell the brain it is learning time
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            # Apply our MixUp trick!
            mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, config.ALPHA, config.DEVICE)

            # Clear old memory
            optimizer.zero_grad()

            # Brain makes a guess
            outputs = model(mixed_images)

            # Calculate how wrong the guess was
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

            # Learn from the mistake
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print the progress at the end of each round (epoch)
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{config.NUM_EPOCHS}], Error Rate (Loss): {avg_loss:.4f}")

    # 4. Save the brain to your computer when finished
    torch.save(model.state_dict(), 'resnet18_mixup.pth')
    print("Training Complete! The AI brain has been saved.")


if __name__ == "__main__":
    train()
