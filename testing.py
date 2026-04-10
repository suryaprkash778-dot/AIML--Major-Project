import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Import from our existing files
import config
from dataset import get_cifar10_dataloaders
from model import get_modified_resnet18


def run_final_exam():
    # 1. Setup Environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load the 10,000 Test Images and the AI Brain
    _, test_loader = get_cifar10_dataloaders()  # We only need the test set now
    model = get_modified_resnet18().to(device)

    # Load your saved weights
    model.load_state_dict(torch.load('resnet18_mixup.pth', map_location=device))

    # ⚠️ CRUCIAL: Put the brain into "Evaluation Mode"
    # This turns off learning and just focuses on guessing
    model.eval()

    # 3. Take the Exam
    correct = 0
    total = 0
    all_predictions = []
    all_actual_labels = []

    print("Starting the Final Exam on 10,000 images... Please wait.")

    # torch.no_grad() tells the computer to stop tracking math for training, saving massive memory
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Brain makes its guess
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # Tally up the score
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Save the answers to draw our graph later
            all_predictions.extend(predicted.cpu().numpy())
            all_actual_labels.extend(labels.cpu().numpy())

    # 4. Calculate Final Grade
    accuracy = 100 * correct / total
    print(f"\n====================================")
    print(f"FINAL EXAM SCORE: {accuracy:.2f}% Accuracy")
    print(f"====================================\n")

    # 5. Draw the IEEE-Ready Graph (Confusion Matrix)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    cm = confusion_matrix(all_actual_labels, all_predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('What the AI Guessed', fontsize=12, fontweight='bold')
    plt.ylabel('The Actual Answer', fontsize=12, fontweight='bold')
    plt.title(f'CIFAR-10 Confusion Matrix\nResNet-18 with MixUp (Accuracy: {accuracy:.2f}%)',
              fontsize=14, fontweight='bold')

    # This will pop the graph open right inside your Colab notebook
    plt.show()


if __name__ == "__main__":
    run_final_exam()
