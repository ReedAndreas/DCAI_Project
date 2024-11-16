import h5py
import numpy as np
from pathlib import Path

print("=== Brain Data Analysis ===\n")

# Load the data
data_path = "brain_data_1000.h5"
print(f"Loading data from: {data_path}")

with h5py.File(data_path, "r") as f:
    brains = f["brains"][:]
    labels = f["labels"][:]

print("\nData Shapes:")
print(f"Brain data shape: {brains.shape}")
print(f"Labels shape: {labels.shape}")

# print the first brain
print(brains[0])

print("\nBrain Data Statistics:")
print(f"Min value: {brains.min():.4f}")
print(f"Max value: {brains.max():.4f}")
print(f"Mean value: {brains.mean():.4f}")
print(f"Std dev: {brains.std():.4f}")

print("\nValue Distribution:")
percentiles = [0, 25, 50, 75, 100]
for p in percentiles:
    print(f"{p}th percentile: {np.percentile(brains, p):.4f}")

print("\nLabel Distribution:")
unique, counts = np.unique(labels, return_counts=True)
for val, count in zip(unique, counts):
    print(f"Label {val}: {count} samples ({count/len(labels)*100:.1f}%)")

print("\nMemory Usage:")
print(f"Brain data: {brains.nbytes / 1024**2:.1f} MB")
print(f"Labels: {labels.nbytes / 1024**2:.1f} MB")
