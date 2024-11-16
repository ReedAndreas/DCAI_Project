import torch
import h5py
import numpy as np
from tqdm import tqdm
from autoencoder import BrainAutoencoder, BrainH5Dataset


def compress_data(model_path, input_h5_path, output_h5_path):
    print("\n=== Starting Data Compression ===")

    # Load the dataset
    dataset = BrainH5Dataset(input_h5_path)
    print(f"Loaded dataset with {len(dataset)} samples")

    # Load labels from original file
    with h5py.File(input_h5_path, "r") as f:
        labels = f["labels"][:]
    print(f"Loaded {len(labels)} labels")

    # Initialize model and load weights
    model = BrainAutoencoder(dataset.data_shape)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Loaded trained autoencoder model")

    # Create arrays to store compressed data
    compressed_data = []

    # Process each brain sample
    print("\nCompressing data...")
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            brain = dataset[i]
            # Add only one dimension for batch
            brain = brain.unsqueeze(0)

            # Get encoded representation
            encoded = model.encoder(brain)
            # Remove the batch dimension
            encoded = encoded.squeeze(0)

            # Store the compressed representation
            compressed_data.append(encoded.numpy())

            # print the shape of the compressed data
            print(f"Compressed data shape: {encoded.shape}")

    # Convert to numpy array
    compressed_data = np.array(compressed_data)

    # Save compressed data and original labels
    print(f"\nSaving compressed data to {output_h5_path}")
    print(f"Compressed shape: {compressed_data.shape}")

    with h5py.File(output_h5_path, "w") as f:
        f.create_dataset("compressed_brains", data=compressed_data)
        f.create_dataset("labels", data=labels)  # Save original labels
        f.attrs["original_shape"] = dataset.data_shape

    # Calculate compression ratio
    original_size = np.prod(dataset.data_shape)
    compressed_size = np.prod(compressed_data.shape[1:])
    compression_ratio = original_size / compressed_size

    print(f"\nCompression complete!")
    print(f"Original size per sample: {original_size:,} values")
    print(f"Compressed size per sample: {compressed_size:,} values")
    print(f"Compression ratio: {compression_ratio:.2f}:1")


def decompress_data(model_path, compressed_h5_path, output_h5_path):
    print("\n=== Starting Data Decompression ===")

    # Load compressed data and labels
    with h5py.File(compressed_h5_path, "r") as f:
        compressed_data = f["compressed_brains"][:]
        labels = f["labels"][:]  # Load labels
        original_shape = f.attrs["original_shape"]

    print(f"Loaded compressed data of shape {compressed_data.shape}")
    print(f"Loaded {len(labels)} labels")

    # Initialize model and load weights
    model = BrainAutoencoder(original_shape)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Loaded trained autoencoder model")

    # Create array for decompressed data
    decompressed_data = []

    # Process each compressed sample
    print("\nDecompressing data...")
    with torch.no_grad():
        for i in tqdm(range(len(compressed_data))):
            # Get compressed representation
            compressed = torch.FloatTensor(compressed_data[i]).unsqueeze(0)

            # Decode
            decoded = model.decoder(compressed)

            # Store the decompressed representation
            decompressed_data.append(decoded.squeeze(0).squeeze(0).numpy())

    # Convert to numpy array
    decompressed_data = np.array(decompressed_data)

    # Save decompressed data and labels
    print(f"\nSaving decompressed data to {output_h5_path}")
    print(f"Decompressed shape: {decompressed_data.shape}")

    with h5py.File(output_h5_path, "w") as f:
        f.create_dataset("brains", data=decompressed_data)
        f.create_dataset("labels", data=labels)  # Save labels with reconstructed data

    print("\nDecompression complete!")


if __name__ == "__main__":
    model_path = "best_autoencoder.pth"
    input_h5_path = "brain_data_1000.h5"
    compressed_h5_path = "brain_data_1000_compressed.h5"
    decompressed_h5_path = "brain_data_1000_decompressed.h5"

    # Compress the data
    compress_data(model_path, input_h5_path, compressed_h5_path)

    # Optionally decompress to verify
    # decompress_data(model_path, compressed_h5_path, decompressed_h5_path)
