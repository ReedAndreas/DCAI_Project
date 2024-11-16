import numpy as np
import torch
import scipy.io
import h5py
import json
import glob
import os
from pathlib import Path
from tqdm import tqdm
import random

# This time we will normalize but the data we will use is predicting who it was based on brain activity


def load_mappings(mappings_dir="mappings"):
    """Load activity and folder mappings."""
    with open(os.path.join(mappings_dir, "level1_mapping.json"), "r") as f:
        activity_labels = json.load(f)

    with open(os.path.join(mappings_dir, "stim_file_mapping.json"), "r") as f:
        folder_mapping = json.load(f)

    # Create folder to activity mapping
    folder_to_activity = {str(v): k for k, v in folder_mapping.items()}
    return activity_labels, folder_to_activity


def get_label_for_file(file_path, folder_to_activity, activity_labels):
    """Get label for a file based on its subject (0-9 for subjects 1-10)."""
    # file name will be of the form sub_<subject_number>_run_<run_number>.mat
    subject_number = int(Path(file_path).stem.split("_")[1])
    if 1 <= subject_number <= 10:  # Only process subjects 1-10
        return subject_number - 1  # Returns 0-9
    return None


def normalize_brain(brain_data, non_zero_mean, non_zero_std):
    """Normalize brain data using scaled Tanh transformation."""
    brain = torch.tensor(brain_data, dtype=torch.float32)

    # Handle NaNs or Infs before normalization
    if torch.isnan(brain).any() or torch.isinf(brain).any():
        print("Warning: NaN or Inf values found in brain data before normalization")
        brain = torch.nan_to_num(brain, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale the data using the non-zero mean and std
    brain_scaled = (brain - non_zero_mean) / non_zero_std

    # Apply the Tanh transformation
    brain_normalized = torch.tanh(brain_scaled)

    return brain_normalized


def process_files(
    mat_files_path="mat_files",
    output_path="brain_data_1000.h5",
    non_zero_mean=None,
    non_zero_std=None,
):
    """Process all .mat files and create h5 dataset."""
    # Load mappings
    activity_labels, folder_to_activity = load_mappings()

    # Get all .mat files for subjects 1-10
    files = []
    for subject in [f"{i:02d}" for i in range(1, 11)]:  # 01 through 10
        subject_pattern = os.path.join(mat_files_path, f"*/sub_{subject}_*.mat")
        files.extend(glob.glob(subject_pattern))

    print(f"Found {len(files)} .mat files for subjects 1-10")

    if len(files) == 0:
        raise ValueError(f"No files found in {mat_files_path} for subjects 1-10")

    # Print first few files for debugging
    print("Sample files found:", files[:5], "..." if len(files) > 5 else "")

    # Shuffle the files
    random.shuffle(files)

    # If non_zero_mean and non_zero_std are not provided, compute them
    if non_zero_mean is None or non_zero_std is None:
        # Collect non-zero voxels from the dataset
        print("Collecting non-zero voxels from the dataset...")
        non_zero_voxels = []
        for file in tqdm(files):
            try:
                mat = scipy.io.loadmat(file)
                brain_data = mat["new_brain"]
                non_zero_voxels.extend(brain_data[brain_data != 0].flatten())
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue

        non_zero_voxels = np.array(non_zero_voxels)

        # Compute mean and std of non-zero voxels
        non_zero_mean = non_zero_voxels.mean()
        non_zero_std = non_zero_voxels.std()

        print(f"Non-zero voxel mean: {non_zero_mean}")
        print(f"Non-zero voxel std: {non_zero_std}")
    else:
        print(f"Using provided non-zero voxel mean: {non_zero_mean}")
        print(f"Using provided non-zero voxel std: {non_zero_std}")

    brains = []
    labels = []
    count = 0

    print("Processing files...")
    count = 0
    for file in tqdm(files):
        try:
            # Load and normalize brain data
            mat = scipy.io.loadmat(file)
            brain = normalize_brain(mat["new_brain"], non_zero_mean, non_zero_std)

            # Get label
            label = get_label_for_file(file, folder_to_activity, activity_labels)
            # only do it for subject 1 and 2

            if label is not None:
                brains.append(brain.numpy())
                labels.append(label)

        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue

        if count >= 2000:
            break

        count += 1

    # Convert to numpy arrays
    brains = np.array(brains)
    labels = np.array(labels, dtype=np.int64)  # Explicitly convert to int64

    print(f"\nProcessed data shape: {brains.shape}")
    print(f"Labels shape: {labels.shape}")
    if len(labels) > 0:
        print(
            f"Label distribution: {np.bincount(labels)}"
        )  # Should show counts for 0-9
    else:
        print("No data was processed!")

    # Save to h5 file
    print(f"\nSaving to {output_path}...")
    with h5py.File(output_path, "w") as f:
        f.create_dataset("brains", data=brains)
        f.create_dataset("labels", data=labels)

    print("Done!")


if __name__ == "__main__":
    process_files(
        mat_files_path="mat_files",
        output_path="brain_data_sub1_to_sub10.h5",  # Updated filename
        non_zero_mean=1341,
        non_zero_std=348,
    )
