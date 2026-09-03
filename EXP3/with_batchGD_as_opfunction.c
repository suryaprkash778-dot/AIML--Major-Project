import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 1. Hardware Device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Training on device: {device}")

# 2. Setup and Data Split
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# BATCH GRADIENT DESCENT: Set batch_size to the entire length of the dataset
# Note: This requires enough GPU VRAM to hold the entire dataset and its gradients at once.
train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=True, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False, pin_memory=True)


# 3. Define the 4-Layer CNN
class CNN4Layer(nn.Module):
    def __init__(self):
        super(CNN4Layer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x


model = CNN4Layer().to(device)

# 4. Optimization and Loss Function
criterion = nn.NLLLoss()
# Standard Gradient Descent (no momentum/nesterov for pure Batch Gradient Descent)
# Learning rate is usually set higher for full batch vs mini-batch
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 5. Training Loop
# Because there is only 1 batch per epoch, we need more epochs to converge
epochs = 50
train_losses, val_losses = [], []

for epoch in range(epochs):
    model.train()
    running_train_loss = 0.0

    # This loop will only run exactly ONCE per epoch because the batch is the whole dataset
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(torch.log(outputs + 1e-9), labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    avg_train_loss = running_train_loss  # Only one batch, so total loss is the average
    train_losses.append(avg_train_loss)

    # Validation Phase
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(torch.log(outputs + 1e-9), labels)
            running_val_loss += loss.item()

    avg_val_loss = running_val_loss
    val_losses.append(avg_val_loss)

    # Print every 5 epochs since we are running 50
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# 6. Plotting Training and Validation Loss
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Training Loss', marker='o', markersize=3)
plt.plot(val_losses, label='Validation Loss', marker='o', markersize=3)
plt.title('Categorical Cross-Entropy Loss over Epochs (Batch GD)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 7. Evaluation Metrics
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(labels.numpy())

print("\nClassification Report:")
print(classification_report(all_targets, all_preds))

cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix on Test Data')
plt.xlabel('Predicted Digit')
plt.ylabel('Actual Digit')
plt.show()
