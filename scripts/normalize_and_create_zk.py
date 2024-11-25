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


def load_mappings(mappings_dir="../mappings"):
    """Load activity and folder mappings."""
    with open(os.path.join(mappings_dir, "level1_mapping.json"), "r") as f:
        activity_labels = json.load(f)

    with open(os.path.join(mappings_dir, "stim_file_mapping.json"), "r") as f:
        folder_mapping = json.load(f)

    # Create folder to activity mapping
    folder_to_activity = {str(v): k for k, v in folder_mapping.items()}
    return activity_labels, folder_to_activity


def get_label_for_file(file_path, folder_to_activity, activity_labels):
    """Get binary label for a file based on its folder."""
    folder_num = str(Path(file_path).parent.name)

    # activity_name = folder_to_activity.get(folder_num)
    # print(folder_num)

    return folder_num
    # if activity_name is None:
    #     print(
    #         f"Warning: No activity mapping found for folder {folder_num} (file: {file_path})"
    #     )
    #     return None

    # label = activity_labels.get(activity_name)
    # if label is None:
    #     print(
    #         f"Warning: No label found for activity {activity_name} (folder: {folder_num})"
    #     )
    #     return None

    # return 1 if label == "S" else 0


def take_difference():
    pass

# ZK don't need to normalize because I did in matlab
def normalize_brain(brain_data, non_zero_mean, non_zero_std):
    """Normalize brain data using scaled Tanh transformation."""
    brain = torch.tensor(brain_data, dtype=torch.float32)
    # diff = abs(brain[:,0] - brain[:,1])
    avg = (brain[:,0] + brain[:,1]) / 2
    sorted_list = sorted(avg)
    # print(avg)
    # print(sorted_list)
    # print(sorted)
    # print("HULLO", int(0.9*len(avg)))
    threshold_idx = int(0.9*len(sorted_list)) - 1
    threshold = sorted_list[threshold_idx]
    threshold_these = np.where(avg <= threshold)
    avg[threshold_these] = 0
    return avg

    # # ZK may just be able to return brain
    # # need to take the difference though


    # # Handle NaNs or Infs before normalization
    # if torch.isnan(brain).any() or torch.isinf(brain).any():
    #     print("Warning: NaN or Inf values found in brain data before normalization")
    #     brain = torch.nan_to_num(brain, nan=0.0, posinf=0.0, neginf=0.0)

    # # Scale the data using the non-zero mean and std
    # # brain_scaled = (brain - non_zero_mean) / non_zero_std
    
    # # Apply the Tanh transformation
    # #brain_normalized = torch.tanh(brain_scaled)

    # #return brain_normalized
    # return brain_scaled


def process_files(
    mat_files_path="mat_files",
    output_path="brain_data_1000.h5",
    non_zero_mean=None,
    non_zero_std=None,
):
    """Process all .mat files and create h5 dataset."""
    # Load mappings
    activity_labels, folder_to_activity = load_mappings()

    # Get all .mat files
    files = glob.glob(os.path.join(mat_files_path, "*/*.mat"))
    print(f"Found {len(files)} .mat files")

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
                # print(mat)
                brain_data = mat["timecourses"]
                non_zero_voxels.extend(brain_data[brain_data != 0].flatten())
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
            break

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
            brain = normalize_brain(mat["timecourses"], non_zero_mean, non_zero_std)
            # Get label
            label = get_label_for_file(file, folder_to_activity, activity_labels)
            if label is not None:
                brains.append(brain.numpy())
                labels.append(label)

        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue

        # if count >= 1000:
        #     break

        count += 1

    # Convert to numpy arrays
    labels = [int(l) for l in labels]
    brains = np.array(brains)
    labels = np.array(labels)
    

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
   # process_files(
   #     mat_files_path="mat_files",
   #     output_path="brain_data_1000.h5",
   #     non_zero_mean=1341,
   #     non_zero_std=348,
   # )
   process_files(
       mat_files_path="/Users/zacharykaras/Desktop/parcel_vectors/",
    #    output_path="/Users/zacharykaras/Desktop/brain_data_all.h5"
        output_path="/Users/zacharykaras/Desktop/brain_data_all_thresholded.h5"
    )

