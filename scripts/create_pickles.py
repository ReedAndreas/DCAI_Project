import numpy as np
import scipy.io as sio
import os
import pickle
from tqdm import tqdm


def create_pickles(mat_files_path, output_path):
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    brains = []
    labels = []

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

        # Get the first (and should be only) mat file in the folder
        mat_files = [f for f in os.listdir(folder_path) if f.endswith(".mat")]
        if not mat_files:
            print(f"Warning: No MAT file found in folder {folder}")
            continue

        mat_file = mat_files[0]
        mat_path = os.path.join(folder_path, mat_file)

        try:
            # Load the MAT file
            mat_data = sio.loadmat(mat_path)

            # Get the first key that contains the brain data
            # Usually it's the last key that's not a metadata key
            data_keys = [k for k in mat_data.keys() if not k.startswith("__")]
            if not data_keys:
                print(f"Warning: No valid data found in {mat_file}")
                continue

            brain_data = mat_data[data_keys[-1]]

            # Add brain data to list
            brains.append(brain_data)

            # Create label (0 for even folders, 1 for odd folders)
            folder_num = int(folder)
            label = 1 if folder_num % 2 else 0
            labels.append(label)

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
    MAT_FILES_PATH = (
        "project_data/mat_files"  # Path to the folder containing numbered folders
    )
    OUTPUT_PATH = "project_data"  # Path where pickle files will be saved

    create_pickles(MAT_FILES_PATH, OUTPUT_PATH)
