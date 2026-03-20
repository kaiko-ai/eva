import numpy as np
import nibabel as nib

from eva.vision.utils.io.nifti import (
    fetch_nifti_axis_direction_code,
    fetch_nifti_shape,
    nifti_to_array,
    read_nifti,
)


# path = "/Users/ioangatop/Desktop/eva/data/flare22/FLARE22Train/images/FLARE22_Tr_0014_0000.nii.gz"
# path = "/Users/ioangatop/Desktop/eva/data/flare22/FLARE22Train/labels/FLARE22_Tr_0014.nii.gz"
# output_path = "./FLARE22_Tr_0014_0000.nii.gz"


# path = "/Users/ioangatop/Desktop/eva/data/flare22/FLARE22Train/images/FLARE22_Tr_0044_0000.nii.gz"
path = "/Users/ioangatop/Desktop/eva/data/flare22/FLARE22Train/labels/FLARE22_Tr_0044.nii.gz"
output_path = "./FLARE22_Tr_0044.nii.gz"


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
