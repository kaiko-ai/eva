import numpy as np

file_path = "tests/eva/assets/vision/datasets/cc_ccii/CC-CCII_public/data/p11-s3161.npy"

arr = np.load(file_path)
print("Original shape:", arr.shape, "dtype:", arr.dtype)

# Decide new target shape for testing
new_shape = (6, 96, 96)  # example shape

# Generate random data of the same dtype
if np.issubdtype(arr.dtype, np.integer):
    arr_small = np.random.randint(
        low=0, high=256,  # full uint8 range
        size=new_shape,
        dtype=arr.dtype
    )
elif np.issubdtype(arr.dtype, np.floating):
    arr_small = np.random.random(new_shape).astype(arr.dtype)
else:
    # fallback: just slice (less common)
    arr_small = arr[tuple(slice(0, s) for s in new_shape)]

print("New shape:", arr_small.shape)

# Overwrite the file with smaller random array
np.save(file_path, arr_small)
