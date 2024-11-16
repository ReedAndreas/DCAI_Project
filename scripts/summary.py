import numpy as np
import scipy.io
from tqdm import tqdm
from pathlib import Path

# Collect all non-zero voxels from the dataset
non_zero_voxels = []
mat_files_path = "mat_files"
files = list(Path(mat_files_path).glob("*/*.mat"))


print("Collecting non-zero voxels from the dataset...")
for file in tqdm(files):
    try:
        mat = scipy.io.loadmat(file)
        brain_data = mat["new_brain"]
        non_zero_voxels.extend(brain_data[brain_data != 0].flatten())
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")
        continue

non_zero_voxels = np.array(non_zero_voxels)

# Compute mean and standard deviation of non-zero voxels
non_zero_mean = non_zero_voxels.mean()
non_zero_std = non_zero_voxels.std()

print(f"Non-zero voxel mean: {non_zero_mean}")
print(f"Non-zero voxel std: {non_zero_std}")
