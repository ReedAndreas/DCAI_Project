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


def load_mappings(mappings_dir="mappings"):
    """Load activity and folder mappings."""
    with open(os.path.join(mappings_dir, "level2_mapping.json"), "r") as f:
        activity_labels = json.load(f)

    with open(os.path.join(mappings_dir, "stim_file_mapping.json"), "r") as f:
        folder_mapping = json.load(f)

    # Create folder to activity mapping
    folder_to_activity = {str(v): k for k, v in folder_mapping.items()}
    return activity_labels, folder_to_activity


def get_label_for_file(file_path, folder_to_activity, activity_labels):
    """Get binary label for a file based on its folder (0 for Ball sport, 1 for Sedentary)."""
    folder_num = str(Path(file_path).parent.name)

    activity_name = folder_to_activity.get(folder_num)
    if activity_name is None:
        print(
            f"Warning: No activity mapping found for folder {folder_num} (file: {file_path})"
        )
        return None

    label = activity_labels.get(activity_name)
    if label is None:
        print(
            f"Warning: No label found for activity {activity_name} (folder: {folder_num})"
        )
        return None

    if label == "Ball sport":
        return 0
    elif label == "Sedentary":
        return 1
    else:
        return None


def normalize_brain(brain_data, non_zero_mean, non_zero_std):
    """Normalize brain data using scaled Tanh transformation."""
    brain = torch.tensor(brain_data, dtype=torch.float32)

    if torch.isnan(brain).any() or torch.isinf(brain).any():
        print("Warning: NaN or Inf values found in brain data before normalization")
        brain = torch.nan_to_num(brain, nan=0.0, posinf=0.0, neginf=0.0)

    brain_scaled = (brain - non_zero_mean) / non_zero_std
    brain_normalized = torch.tanh(brain_scaled)

    return brain_normalized


def process_files(
    mat_files_path="mat_files",
    output_path="brain_data_1000.h5",
    non_zero_mean=None,
    non_zero_std=None,
    samples_per_class=200,  # Number of samples to collect per class
):
    """Process .mat files and create balanced h5 dataset for subjects 1-3."""
    activity_labels, folder_to_activity = load_mappings()

    # Get files for subjects 1-3
    files = []
    for subject in ["01", "02", "03"]:
        subject_pattern = os.path.join(mat_files_path, f"*/sub_{subject}_*.mat")
        files.extend(glob.glob(subject_pattern))

    print(f"Found {len(files)} .mat files for subjects 1-3")

    if len(files) == 0:
        raise ValueError(f"No files found for subjects 1-3 in {mat_files_path}")

    # Print first few files for debugging
    print("Sample files found:", files[:5], "..." if len(files) > 5 else "")

    random.shuffle(files)

    # Handle normalization parameters
    if non_zero_mean is None or non_zero_std is None:
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
        non_zero_mean = non_zero_voxels.mean()
        non_zero_std = non_zero_voxels.std()
        print(f"Non-zero voxel mean: {non_zero_mean}")
        print(f"Non-zero voxel std: {non_zero_std}")
    else:
        print(f"Using provided non-zero voxel mean: {non_zero_mean}")
        print(f"Using provided non-zero voxel std: {non_zero_std}")

    # Separate collection for each class
    ball_sport_data = []
    sedentary_data = []

    print("Processing files...")
    for file in tqdm(files):
        try:
            # Get label first to avoid unnecessary processing
            label = get_label_for_file(file, folder_to_activity, activity_labels)

            if label is not None:
                # Only process brain data if we need more samples for this class
                if (label == 0 and len(ball_sport_data) < samples_per_class) or (
                    label == 1 and len(sedentary_data) < samples_per_class
                ):

                    # Load and normalize brain data
                    mat = scipy.io.loadmat(file)
                    brain = normalize_brain(
                        mat["new_brain"], non_zero_mean, non_zero_std
                    )

                    if label == 0:
                        ball_sport_data.append(brain.numpy())
                    else:
                        sedentary_data.append(brain.numpy())

                    # Print progress
                    print(
                        f"\rBall sport: {len(ball_sport_data)}/{samples_per_class}, "
                        f"Sedentary: {len(sedentary_data)}/{samples_per_class}",
                        end="",
                    )

            # Check if we have enough samples for both classes
            if (
                len(ball_sport_data) >= samples_per_class
                and len(sedentary_data) >= samples_per_class
            ):
                break

        except Exception as e:
            print(f"\nError processing {file}: {str(e)}")
            continue

    print("\n")  # New line after progress

    # Ensure we have equal numbers
    ball_sport_data = ball_sport_data[:samples_per_class]
    sedentary_data = sedentary_data[:samples_per_class]

    # Combine the data
    brains = np.array(ball_sport_data + sedentary_data)
    labels = np.array(
        [0] * len(ball_sport_data) + [1] * len(sedentary_data), dtype=np.int64
    )

    # Shuffle the combined data
    shuffle_idx = np.random.permutation(len(brains))
    brains = brains[shuffle_idx]
    labels = labels[shuffle_idx]

    print(f"\nProcessed data shape: {brains.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label distribution: {np.bincount(labels)}")

    # Save to h5 file
    print(f"\nSaving to {output_path}...")
    with h5py.File(output_path, "w") as f:
        f.create_dataset("brains", data=brains)
        f.create_dataset("labels", data=labels)

    print("Done!")


if __name__ == "__main__":
    process_files(
        mat_files_path="mat_files",
        output_path="brain_data_sub1to3_balanced_sport.h5",
        non_zero_mean=1341,
        non_zero_std=348,
        samples_per_class=200,  # Will result in 1000 total samples, balanced between classes
    )
