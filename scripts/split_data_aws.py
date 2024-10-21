# basically we need to follow EDA.ipynb to split the data into runs


import numpy as np
import pandas as pd
import nibabel as nib
import os
import json

base_data_dir = "3892_data/"
TR = 2.0


def split_nifti(input_file, output_file, start_time, end_time, tr):
    # Load the NIfTI file
    img = nib.load(input_file)
    data = img.get_fdata()

    # Calculate start and end volumes
    start_vol = int(start_time / tr)
    end_vol = int(end_time / tr)

    # Extract the desired time points
    split_data = data[:, :, :, start_vol:end_vol]

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
        "task_data",
        subject_dir,
        "events",
        f"{subject_dir}_ses-action01_task-action_run-{run:02d}_events.tsv",  # keep .02d
    )
    return pd.read_csv(events_file, sep="\t")


def format_events_df(events_df, id_to_stim_file):
    events_df["stim_file"] = events_df["stim_file"].str.split("/").str[0]
    events_df["id"] = events_df["stim_file"].map(id_to_stim_file)
    events_df["end_time"] = events_df["onset"] + events_df["duration"]
    return events_df


def read_mappings():
    mapping_file = "/home/ubuntu/DCAI_Project/mappings/id_to_stim_file.json"
    with open(mapping_file, "r") as f:
        id_to_stim_file = json.load(f)
    return id_to_stim_file


class StimulusMapper:
    def __init__(self, mapping_file):
        self.mapping_file = mapping_file
        self.mapping = self.load_mapping()
        self.next_id = max(self.mapping.values(), default=0) + 1

    def load_mapping(self):
        if os.path.exists(self.mapping_file):
            with open(self.mapping_file, "r") as f:
                return json.load(f)
        return {}

    def save_mapping(self):
        with open(self.mapping_file, "w") as f:
            json.dump(self.mapping, f, indent=2)

    def get_or_create_id(self, stim_file):
        if stim_file not in self.mapping:
            self.mapping[stim_file] = self.next_id
            self.next_id += 1
        return self.mapping[stim_file]


def process_subject(subject_dir, mapper):
    # Unzip data if needed
    unzip_data(os.path.join(base_data_dir, "clean_data", subject_dir))

    # Iterate through runs
    for run in range(1, 13):  # Assuming 12 runs, adjust if needed
        # Read and format events file for this run
        events_df = read_events_file(subject_dir, run)
        events_df["stim_file"] = events_df["stim_file"].str.split("/").str[0]
        events_df["id"] = events_df["stim_file"].apply(mapper.get_or_create_id)
        events_df["end_time"] = events_df["onset"] + events_df["duration"]

        nifti_file = os.path.join(
            base_data_dir,
            "clean_data",
            subject_dir,
            f"clean_sub-{subject_dir[-2:]}_task-action_run-{run}_desc-preproc_bold.nii",  # should not be .02d
        )

        # Iterate through events and split NIFTI files
        for _, row in events_df.iterrows():
            output_dir = os.path.join(base_data_dir, "split_data", str(row["id"]))
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(
                output_dir,
                f"sub_{subject_dir[-2:]}_run_{run:02d}.nii",
            )

            split_nifti(nifti_file, output_file, row["onset"], row["end_time"], TR)

    return events_df


def main():
    mapping_file = os.path.join(base_data_dir, "mappings", "stim_file_mapping.json")
    log_file = os.path.join(base_data_dir, "logs", "split_data_log.txt")
    mapper = StimulusMapper(mapping_file)

    # Iterate through subject directories
    for subject_num in range(11, 31):  # Start from 01 and go to 30
        subject_dir = f"sub-{subject_num:02d}"
        subject_path = os.path.join(base_data_dir, "clean_data", subject_dir)
        if os.path.exists(subject_path):
            try:
                events_df = process_subject(subject_dir, mapper)
                print(f"Processed {subject_dir}")
                print(events_df)
            except Exception as e:
                print(f"Error processing {subject_dir}: {str(e)}")
        else:
            print(f"Subject directory {subject_dir} not found")

        # append to log file the subject that was processed
        with open(log_file, "a") as f:
            f.write(f"{subject_dir}\n")

    # Save updated mappings
    mapper.save_mapping()


if __name__ == "__main__":
    main()
