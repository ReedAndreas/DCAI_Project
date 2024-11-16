import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import h5py
from tqdm import tqdm


# Dataset class for loading brain data
class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_file_path):
        self.h5_file = h5py.File(h5_file_path, "r")
        self.brains = self.h5_file["brains"]

    def __len__(self):
        return len(self.brains)

    def __getitem__(self, idx):
        brain = torch.tensor(self.brains[idx], dtype=torch.float32)
        # print(f"Loaded brain shape before unsqueeze: {brain.shape}")
        brain = brain.unsqueeze(0)  # Add channel dimension
        # print(f"Loaded brain shape after unsqueeze: {brain.shape}")
        return brain

    def __del__(self):
        self.h5_file.close()


# Define the autoencoder
import torch
import torch.nn as nn


class Autoencoder3D(nn.Module):
    def __init__(self):
        super(Autoencoder3D, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),  # Halves dimensions
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),  # Halves dimensions
            nn.Conv3d(32, 1, kernel_size=3, stride=2, padding=1),  # Halves dimensions
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                1, 32, kernel_size=4, stride=2, padding=1
            ),  # Doubles dimensions
            nn.ReLU(),
            nn.ConvTranspose3d(
                32, 16, kernel_size=4, stride=2, padding=1
            ),  # Doubles dimensions
            nn.ReLU(),
            nn.ConvTranspose3d(
                16, 1, kernel_size=4, stride=2, padding=1
            ),  # Doubles dimensions
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # Ensure output size matches input size
        if decoded.size() != x.size():
            decoded = F.interpolate(
                decoded, size=x.size()[2:], mode="trilinear", align_corners=False
            )
        return decoded


# Validate dimensions programmatically
def validate_dimensions():
    model = Autoencoder3D()
    test_input = torch.randn(1, 1, 91, 109, 91)  # Batch size = 1, single channel
    output = model(test_input)
    # print(f"Input shape: {test_input.shape}")
    # print(f"Output shape: {output.shape}")
    assert output.shape == test_input.shape, "Output dimensions do not match input!"


# Set random seed for reproducibility
def set_random_seed(seed=42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Main script
if __name__ == "__main__":
    # Set random seed
    set_random_seed()
    # validate_dimensions()

    # Load dataset and create train/validation splits
    dataset = HDF5Dataset("brain_data_1000.h5")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Initialize autoencoder and move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = Autoencoder3D().to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Reconstruction loss
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 50
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        autoencoder.train()
        train_loss = 0.0

        # Training step
        for brains in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            brains = brains.to(device)
            optimizer.zero_grad()
            outputs = autoencoder(brains)
            loss = criterion(outputs, brains)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation step
        autoencoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for brains in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                brains = brains.to(device)
                outputs = autoencoder(brains)
                loss = criterion(outputs, brains)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Save the latest model
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": autoencoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            f"autoencoder3d_epoch_{epoch+1}.pth",
        )

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": autoencoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                "autoencoder3d_best.pth",
            )

        print("-" * 50)
