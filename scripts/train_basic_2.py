from tqdm import tqdm
import time
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import torch

import random
import numpy as np
import torch


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable auto-tuning for repeatability


class HDF5Dataset(Dataset):
    def __init__(self, h5_file_path):
        self.h5_file = h5py.File(h5_file_path, "r")
        self.brains = self.h5_file["brains"]
        self.labels = self.h5_file["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        brain = torch.tensor(self.brains[idx], dtype=torch.float32)
        brain = brain.unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return brain, label

    def __del__(self):
        self.h5_file.close()


# Load dataset
full_dataset = HDF5Dataset("brain_data_1000.h5")

# Split dataset into training and validation sets
train_size = int(0.8 * len(full_dataset))  # 80% training
val_size = len(full_dataset) - train_size  # 20% validation
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

from monai.networks.nets import DenseNet121
import torch.nn as nn


# Add this helper function at the top of your script
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Metal) device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device


# Add diagnostic info after imports
print("\n=== PyTorch Device Information ===")
print(f"PyTorch version: {torch.__version__}")
print(f"MPS (Apple Metal) available: {torch.backends.mps.is_available()}")
print(f"CUDA available: {torch.cuda.is_available()}")
device = get_device()
print(f"Selected device: {device}")
print("================================\n")

# Modify the model initialization
model = DenseNet121(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
).to(device)

model.class_layers.out = nn.Linear(
    in_features=model.class_layers.out.in_features, out_features=2
).to(device)

import torch.optim as optim
import torch.nn.functional as F

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    set_random_seed(64)
    device = get_device()
    model.train()

    first_batch_done = False
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs")

    for epoch in epoch_pbar:
        running_loss = 0.0
        correct = 0
        total = 0

        batch_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False, unit="batch"
        )

        for i, (images, labels) in enumerate(batch_pbar):
            # Move data to device
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
        print("-" * 50)

        epoch_pbar.set_postfix(
            {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}
        )


# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50)
