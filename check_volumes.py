import nibabel as nib
import sys
import os


def check_volumes(nifti_path):
    """Check the number of volumes in a NIFTI file."""
    try:
        img = nib.load(nifti_path)
        shape = img.shape

        print(f"\nFile: {os.path.basename(nifti_path)}")
        print(f"Shape: {shape}")
        if len(shape) == 4:
            print(f"Number of volumes: {shape[3]}")
        else:
            print("This appears to be a 3D image with no time dimension")

    except Exception as e:
        print(f"Error loading {nifti_path}: {str(e)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_volumes.py path/to/file.nii[.gz]")
    else:
        nifti_path = sys.argv[1]
        check_volumes(nifti_path)
