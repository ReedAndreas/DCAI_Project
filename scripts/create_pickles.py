import numpy as np
import scipy.io as sio
import os
import pickle
import json
from tqdm import tqdm


def create_pickles(mat_files_path, output_path):
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    brains = []
    labels = []

    # Load both mapping files
    with open("mappings/level1_mapping.json", "r") as f:
        activity_labels = json.load(f)  # Maps activity names to S/NS

    with open("mappings/stim_file_mapping.json", "r") as f:
        folder_mapping = json.load(f)  # Maps activity names to folder numbers

    # Create reverse mapping from folder numbers to activity names
    folder_to_activity = {str(v): k for k, v in folder_mapping.items()}

    # Get all folders (1-180)
    folders = sorted(
        [
            f
            for f in os.listdir(mat_files_path)
            if os.path.isdir(os.path.join(mat_files_path, f))
        ],
        key=int,
    )

    print("Processing MAT files...")
    for folder in tqdm(folders):
        folder_path = os.path.join(mat_files_path, folder)

        # Get activity name for this folder
        activity_name = folder_to_activity.get(folder)
        if activity_name is None:
            print(f"Warning: No activity mapping found for folder {folder}")
            continue

        # Get label (S/NS) for this activity
        label = activity_labels.get(activity_name)
        if label is None:
            print(f"Warning: No label found for activity {activity_name}")
            continue

        # Convert S/NS to binary
        binary_label = 1 if label == "S" else 0

        # Get all mat files in the folder
        mat_files = [f for f in os.listdir(folder_path) if f.endswith(".mat")]
        if not mat_files:
            print(f"Warning: No MAT file found in folder {folder}")
            continue

        for mat_file in mat_files:
            mat_path = os.path.join(folder_path, mat_file)

            try:
                # Load the MAT file
                mat_data = sio.loadmat(mat_path)

                # Get the first key that contains the brain data
                data_keys = [k for k in mat_data.keys() if not k.startswith("__")]
                if not data_keys:
                    print(f"Warning: No valid data found in {mat_file}")
                    continue

                brain_data = mat_data[data_keys[-1]]

                # Add brain data and label to lists
                brains.append(brain_data)
                labels.append(binary_label)

            except Exception as e:
                print(f"Error processing {mat_file}: {str(e)}")
                continue

    # Convert lists to numpy arrays
    brains = np.array(brains)
    labels = np.array(labels)

    print(f"\nFinal shapes:")
    print(f"Brains: {brains.shape}")
    print(f"Labels: {labels.shape}")

    # Save to pickle files
    print("\nSaving pickle files...")
    with open(os.path.join(output_path, "brains.pickle"), "wb") as f:
        pickle.dump(brains, f)

    with open(os.path.join(output_path, "labels.pickle"), "wb") as f:
        pickle.dump(labels, f)

    print("Done! Files saved to:")
    print(f"- {os.path.join(output_path, 'brains.pickle')}")
    print(f"- {os.path.join(output_path, 'labels.pickle')}")


if __name__ == "__main__":
    # Set your paths here
    MAT_FILES_PATH = "./sub_mat_files"  # Path to the folder containing numbered folders
    OUTPUT_PATH = "./"  # Path where pickle files will be saved

    create_pickles(MAT_FILES_PATH, OUTPUT_PATH)
