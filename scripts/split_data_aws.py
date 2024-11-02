# basically we need to follow EDA.ipynb to split the data into runs


import numpy as np
import pandas as pd
import nibabel as nib
import os
import json
import argparse

base_data_dir = "3892_data/"
TR = 2.0


def split_nifti(input_file, output_file, volume_indices):
    # Load the NIfTI file
    img = nib.load(input_file)
    data = img.get_fdata()

    # Extract the desired volumes
    split_data = data[:, :, :, volume_indices]

    # Create a new NIfTI image with the split data
    split_img = nib.Nifti1Image(split_data, img.affine, img.header)

    # Save the new image
    nib.save(split_img, output_file)
    print(f"Split NIfTI saved: {output_file}")


def unzip_data(curr_sub_dir):
    for file in os.listdir(curr_sub_dir):
        if file.endswith(".gz"):
            unzipped_file = os.path.join(curr_sub_dir, file[:-3])
            # Only unzip if the unzipped file doesn't already exist
            if not os.path.exists(unzipped_file):
                os.system(f"gunzip -f {os.path.join(curr_sub_dir, file)}")


def read_events_file(subject_dir, run):
    events_file = os.path.join(
        base_data_dir,
        "conditions_by_volume",
        subject_dir,
        f"{subject_dir}_ses-action01_task-action_run-{run:02d}_events.tsv",
    )
    return pd.read_csv(
        events_file,
        sep="\t",
        header=None,
        names=["volume", "index", "time", "condition", "condition_id"],
    )


def process_subject(subject_dir, dry_run=False):
    # Unzip data if needed
    unzip_data(os.path.join(base_data_dir, "clean_data", subject_dir))

    # Iterate through runs
    for run in range(1, 13):
        # Read events file for this run
        events_df = read_events_file(subject_dir, run)

        # Group consecutive non-Rest volumes by condition
        current_condition = None
        current_volumes = []
        current_id = None

        nifti_file = os.path.join(
            base_data_dir,
            "clean_data",
            subject_dir,
            f"clean_sub-{subject_dir[-2:]}_task-action_run-{run}_desc-preproc_bold.nii",
        )

        # Process each volume
        for _, row in events_df.iterrows():
            if row["condition"] != "Rest":
                if current_condition != row["condition"]:
                    # Save previous group if it exists
                    if current_volumes:
                        output_dir = os.path.join(
                            base_data_dir, "split_data", str(int(current_id))
                        )
                        os.makedirs(output_dir, exist_ok=True)
                        output_file = os.path.join(
                            output_dir,
                            f"sub_{subject_dir[-2:]}_run_{run:02d}.nii",
                        )
                        split_nifti(nifti_file, output_file, current_volumes)

                    # Start new group
                    current_condition = row["condition"]
                    current_volumes = [row["volume"]]
                    current_id = row["condition_id"]
                else:
                    current_volumes.append(row["volume"])

        # Save last group if it exists
        if current_volumes:
            output_dir = os.path.join(base_data_dir, "split_data", str(int(current_id)))
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(
                output_dir,
                f"sub_{subject_dir[-2:]}_run_{run:02d}.nii",
            )
            split_nifti(nifti_file, output_file, current_volumes)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Split NIFTI data based on conditions")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without actually splitting files",
    )
    parser.add_argument(
        "--test-subject",
        action="store_true",
        help="Only process one subject (sub-11) for testing",
    )
    args = parser.parse_args()

    log_file = os.path.join(base_data_dir, "logs", "split_data_log.txt")

    if args.test_subject:
        # Process just one subject
        subject_dir = "sub-11"
        process_subject(subject_dir, dry_run=args.dry_run)
    else:
        # Process all subjects
        for subject_num in range(0, 31):
            subject_dir = f"sub-{subject_num:02d}"
            subject_path = os.path.join(base_data_dir, "clean_data", subject_dir)
            if os.path.exists(subject_path):
                try:
                    process_subject(subject_dir, dry_run=args.dry_run)
                    print(f"Processed {subject_dir}")
                except Exception as e:
                    print(f"Error processing {subject_dir}: {str(e)}")
            else:
                print(f"Subject directory {subject_dir} not found")

            # append to log file only if not in dry run mode
            if not args.dry_run:
                with open(log_file, "a") as f:
                    f.write(f"{subject_dir}\n")


if __name__ == "__main__":
    main()
