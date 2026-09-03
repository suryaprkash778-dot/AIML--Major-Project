import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# 1. Setup and Data Split (Allocating 20% for Validation)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)


# 2. Define the 4-Layer CNN
class CNN4Layer(nn.Module):
    def __init__(self):
        super(CNN4Layer, self).__init__()
        # Layer 1: Convolutional
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 2: Convolutional
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Layer 3: Fully Connected
        # After two 2x2 poolings, the 28x28 image becomes 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)

        # Layer 4: Output Layer
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()
        # Explicit Softmax activation for the output layer
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x


model = CNN4Layer()

# 3. Optimization and Loss Function
# Note on PyTorch: nn.CrossEntropyLoss() applies Softmax internally. Since we explicitly
# included Softmax in our model architecture to meet the requirement, applying it again
# causes mathematical errors. Instead, we take the log of our Softmax outputs and use
# nn.NLLLoss(), which mathematically equals Categorical Cross-Entropy (CCE).
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)

# 4. Training Loop with Validation Tracking
epochs = 5
train_losses, val_losses = [], []

for epoch in range(epochs):
    model.train()
    running_train_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        # Apply log to Softmax probabilities for NLLLoss
        loss = criterion(torch.log(outputs + 1e-9), labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation Phase
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(torch.log(outputs + 1e-9), labels)
            running_val_loss += loss.item()

    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# 5. Plotting Training and Validation Loss
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Training Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='o')
plt.title('Categorical Cross-Entropy Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 6. Evaluation Metrics: Confusion Matrix and Classification Report
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_targets.extend(labels.numpy())

print("\nClassification Report:")
print(classification_report(all_targets, all_preds))

# Plotting the Confusion Matrix
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix on Test Data')
plt.xlabel('Predicted Digit')
plt.ylabel('Actual Digit')
plt.show()
