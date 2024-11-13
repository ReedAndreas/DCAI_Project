import os
import shutil

# Basically I want to write the first .mat file in every folder in mat_files/* to a new folder called sub_mat_files/*

mat_files_path = "mat_files"
sub_mat_files_path = "sub_mat_files"

# Get all folders in mat_files
folders = sorted(
    [
        f
        for f in os.listdir(mat_files_path)
        if os.path.isdir(os.path.join(mat_files_path, f))
    ],
    key=int,
)

for folder in folders:
    folder_path = os.path.join(mat_files_path, folder)

    # Create corresponding folder in sub_mat_files
    sub_folder_path = os.path.join(sub_mat_files_path, folder)
    os.makedirs(sub_folder_path, exist_ok=True)

    # Get first 4 .mat files in folder
    mat_files = [f for f in os.listdir(folder_path) if f.endswith(".mat")]
    if len(mat_files) < 15:
        print(f"Warning: No MAT file found in folder {folder}")
        continue

    for mat_file in mat_files[:15]:
        src_path = os.path.join(folder_path, mat_file)

        # Copy to same folder structure in sub_mat_files
        dst_path = os.path.join(sub_folder_path, mat_file)
        shutil.copy2(src_path, dst_path)

print("Done! Files copied to sub_mat_files/")
