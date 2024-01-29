# from eva.vision.data import datasets

# dataset = datasets.PatchCamelyon(root="./data/patch_camelyon", split="val")
# dataset.setup()
# image, target = dataset[0]

import h5py
import gzip
import io

path = "data/patch_camelyon/camelyon/patch_level_2_split_valid_x.h5"
path_h5 = "camelyonpatch_level_2_split_valid_x.h5"
path_gz = "camelyonpatch_level_2_split_valid_x.h5.gz"

# wget -c https://zenodo.org/records/2546921/files/camelyonpatch_level_2_split_test_x.h5.gz?download=1 -O - | tar -xz

# wget -c https://zenodo.org/records/2546921/files/camelyonpatch_level_2_split_test_y.h5.gz?download=1 -O - | tar -xz

with h5py.File(path_h5, "r") as file:
    images = file.get("x")[()]

# with gzip.open(path, "rb") as gz_file:
#     with io.BytesIO(gz_file.read()) as h5_gzipped:
#         with h5py.File(h5_gzipped, "r") as file:
#             images = file.get("x")[()]

# print(images.shape)
# print("The memory size of numpy array arr is:",images.itemsize*images.size,"bytes")


# pdm run python main.py  12.46s user 3.25s system 118% cpu 13.310 total
            
# for _ in range(5):
#     # print(images.get("x"))
#     print(images)

