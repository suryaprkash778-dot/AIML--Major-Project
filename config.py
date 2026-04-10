# config.py
import torch

# Data configurations
BATCH_SIZE = 128
NUM_WORKERS = 2  # Adjust based on your CPU cores (2 or 4 is usually safe)
NUM_CLASSES = 10

# Training hyperparameters
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

# MixUp / CutMix Hyperparameters
ALPHA = 1.0

# Device configuration (Automatically uses GPU if available, else CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
