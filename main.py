import numpy as np
import nibabel as nib

from eva.vision.utils.io.nifti import (
    fetch_nifti_axis_direction_code,
    fetch_nifti_shape,
    nifti_to_array,
    read_nifti,
)

# path = "/Users/ioangatop/Desktop/eva/data/word/WORD-V0.1.0/imagesTr/word_0051.nii.gz"
# path = "/Users/ioangatop/Desktop/eva/data/word/WORD-V0.1.0/labelsTr/word_0051.nii.gz"
# output_path = "./word_0051.nii.gz"

# path = "/Users/ioangatop/Desktop/eva/data/word/WORD-V0.1.0/imagesTr/word_0130.nii.gz"
# path = "/Users/ioangatop/Desktop/eva/data/word/WORD-V0.1.0/labelsTr/word_0130.nii.gz"
# output_path = "./word_0130.nii.gz"


# path = "/Users/ioangatop/Desktop/eva/data/word/WORD-V0.1.0/imagesVal/word_0001.nii.gz"
path = "/Users/ioangatop/Desktop/eva/data/word/WORD-V0.1.0/labelsVal/word_0001.nii.gz"
output_path = "./word_0001.nii.gz"


volume = read_nifti(path=path)

array = nifti_to_array(volume)
print("Original shape:", array.shape)

x, y, z = array.shape

cropped = array[
    x//4 : 3*x//8,
    y//4 : 3*y//8,
    z//4 : 3*z//8
]

print("Cropped shape:", cropped.shape)


affine = volume.affine
header = volume.header

cropped_nifti = nib.Nifti1Image(cropped, affine, header)

nib.save(cropped_nifti, output_path)

print(f"Saved cropped file to: {output_path}")