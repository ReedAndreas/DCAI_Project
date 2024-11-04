import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import pickle

# load in the data
with open("../project_data/brains.pickle", "rb") as f:
    new_brains = pickle.load(f)
with open("../project_data/labels.pickle", "rb") as f:
    labels = pickle.load(f)


import torch
from torch.utils.data import DataLoader, Dataset


# Custom dataset for 3D brain data
class BrainDataset(Dataset):
    def __init__(self, data, labels):
        # Convert numpy arrays to torch tensors
        self.data = [torch.from_numpy(arr).float() for arr in data]
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]  # Already a tensor
        label = self.labels[idx]
        # Add channel dimension
        image = image.unsqueeze(0)  # Shape: [1, depth, height, width]
        return image, torch.tensor(label, dtype=torch.long)


# Create dataset and dataloader
train_dataset = BrainDataset(new_brains, labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

import torch.optim as optim
import torch.nn.functional as F

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()

    # Time the first batch to estimate epoch duration
    first_batch_done = False

    # Create epoch progress bar
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs")

    for epoch in epoch_pbar:
        running_loss = 0.0
        correct = 0
        total = 0

        # Create batch progress bar with additional metrics
        batch_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}", leave=False, unit="batch", miniters=1
        )  # Update at every iteration

        for i, (images, labels) in enumerate(batch_pbar):
            batch_start_time = time.time()

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate batch time
            batch_time = time.time() - batch_start_time

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update running statistics
            running_loss += loss.item()
            current_loss = running_loss / (i + 1)
            current_acc = 100 * correct / total

            # Estimate time for first batch
            if not first_batch_done:
                estimated_epoch_time = batch_time * len(train_loader)
                print(f"\nEstimated time per epoch: {estimated_epoch_time:.2f} seconds")
                print(
                    f"Estimated total training time: {estimated_epoch_time * num_epochs:.2f} seconds"
                )
                first_batch_done = True

            # Update batch progress bar
            batch_pbar.set_postfix(
                {
                    "loss": f"{current_loss:.4f}",
                    "acc": f"{current_acc:.2f}%",
                    "batch_time": f"{batch_time:.2f}s",
                    "memory": (
                        f"{torch.cuda.memory_allocated() / 1e9:.1f}GB"
                        if torch.cuda.is_available()
                        else "N/A"
                    ),
                }
            )

            # Optional: Print every N batches
            if (i + 1) % 10 == 0:  # Adjust frequency as needed
                print(
                    f"\nBatch {i+1}/{len(train_loader)}: "
                    f"Loss: {current_loss:.4f}, "
                    f"Accuracy: {current_acc:.2f}%, "
                    f"Time/batch: {batch_time:.2f}s"
                )

        # Update epoch progress bar
        epoch_pbar.set_postfix(
            {"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.2f}%"}
        )

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Average Loss: {current_loss:.4f}")
        print(f"Accuracy: {current_acc:.2f}%")
        print("-" * 50)


# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=10)