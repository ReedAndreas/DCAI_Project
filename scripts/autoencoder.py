import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import os
import numpy as np
import torch.nn.functional as F


class BrainDataset(Dataset):
    def __init__(self, root_dir, max_samples_per_epoch=1000):
        self.root_dir = root_dir
        self.file_paths = []
        print(f"\nInitializing BrainDataset from: {root_dir}")

        # Collect all .mat files from folders 1-180
        total_files = 0
        for folder in range(1, 181):
            folder_path = os.path.join(root_dir, str(folder))
            if os.path.exists(folder_path):
                mat_files = [
                    os.path.join(folder_path, f)
                    for f in os.listdir(folder_path)
                    if f.endswith(".mat")
                ]
                self.file_paths.extend(mat_files)
                total_files += len(mat_files)
                if folder % 20 == 0:  # Print progress every 20 folders
                    print(
                        f"Processed folder {folder}/180, found {total_files} files so far..."
                    )

        # Randomly subsample files if there are too many
        if len(self.file_paths) > max_samples_per_epoch:
            self.file_paths = np.random.choice(
                self.file_paths, max_samples_per_epoch, replace=False
            ).tolist()
            print(f"Randomly sampled {max_samples_per_epoch} files for this epoch")

        print(f"Dataset initialization complete. Using {len(self.file_paths)} files")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        mat_path = self.file_paths[idx]
        try:
            mat_data = sio.loadmat(mat_path)
            brain_data = mat_data["new_brain"]
            brain_tensor = torch.FloatTensor(brain_data)
            if idx % 100 == 0:  # Log every 100th item
                print(
                    f"Loaded item {idx}, shape: {brain_tensor.shape}, from: {mat_path}"
                )
            return brain_tensor
        except Exception as e:
            print(f"Error loading file {mat_path}: {str(e)}")
            raise


class BrainAutoencoder(nn.Module):
    def __init__(self, input_shape):
        super(BrainAutoencoder, self).__init__()

        # Calculate proper padding
        self.input_shape = (1,) + input_shape  # (1, 91, 109, 91)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                64, 32, kernel_size=4, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose3d(
                32, 16, kernel_size=4, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose3d(
                16, 1, kernel_size=4, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Add channel dimension if not present
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension
            x = x.unsqueeze(0)  # Add channel dimension
        elif len(x.shape) == 4:
            x = x.unsqueeze(1)  # Add channel dimension only

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        # Upsample to match input size
        decoded = F.interpolate(
            decoded, size=x.shape[2:], mode="trilinear", align_corners=False
        )
        return decoded.squeeze(1)  # Remove channel dimension in output


# Add training function
def train_autoencoder(model, dataset, num_epochs=50, batch_size=2, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on device: {device}")
    print(
        f"Training parameters: batch_size={batch_size}, learning_rate={learning_rate}, num_epochs={num_epochs}"
    )

    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("\nStarting training...")
    for epoch in range(num_epochs):
        # Create a new dataset for each epoch with random sampling
        epoch_dataset = BrainDataset(dataset.root_dir, max_samples_per_epoch=1000)
        dataloader = DataLoader(epoch_dataset, batch_size=batch_size, shuffle=True)
        print(f"\nEpoch {epoch+1}: Created DataLoader with {len(dataloader)} batches")

        total_loss = 0
        batch_count = 0

        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)

            if batch_idx == 0:
                print(f"Input batch shape: {batch.shape}")

            outputs = model(batch)
            loss = criterion(outputs, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            if batch_idx % 10 == 0:
                print(
                    f"Batch {batch_idx}/{len(dataloader)}, "
                    f"Current loss: {loss.item():.6f}"
                )

        avg_loss = total_loss / batch_count
        print(
            f"\nEpoch [{epoch+1}/{num_epochs}] Complete - Average Loss: {avg_loss:.6f}"
        )

        # Save model checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"autoencoder_checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    root_dir = "./mat_files"
    print("\n=== Starting Brain Autoencoder Training ===")
    print(f"Looking for data in: {root_dir}")

    # Create dataset
    dataset = BrainDataset(root_dir)
    input_shape = dataset[0].shape
    print(f"\nDataset created successfully:")
    print(f"- Total samples: {len(dataset)}")
    print(f"- Sample shape: {input_shape}")

    # Create model
    print("\nInitializing model...")
    model = BrainAutoencoder(input_shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} total parameters")

    # Print model architecture
    print("\nModel Architecture:")
    print(model)

    # Train model
    print("\nStarting training process...")
    train_autoencoder(model, dataset)

    print("\n=== Training Complete ===")
