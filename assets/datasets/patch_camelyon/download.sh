#!/bin/sh
# ----------------------------------------------
# Downloads and unzips the PatchCamelyon dataset
# ----------------------------------------------

# The local path to download the data
OUTPUT_DIR="${OUTPUT_DIR:-data/patch_camelyon}" 

declare -a FILENAMES=(
    "camelyonpatch_level_2_split_train_x.h5.gz"
    "camelyonpatch_level_2_split_train_y.h5.gz"
    "camelyonpatch_level_2_split_train_meta.csv"
    "camelyonpatch_level_2_split_valid_x.h5.gz"
    "camelyonpatch_level_2_split_valid_y.h5.gz"
    "camelyonpatch_level_2_split_valid_meta.csv"
    "camelyonpatch_level_2_split_test_x.h5.gz"
    "camelyonpatch_level_2_split_test_y.h5.gz"
    "camelyonpatch_level_2_split_test_meta.csv"
)
for FILE in "${FILENAMES[@]}"
do
    OUTPUT_FILE=${OUTPUT_DIR}/${FILE}
    if [ -f ${OUTPUT_FILE%%.gz} ] ; then
        continue
    fi
    mkdir -m 777 -p ${OUTPUT_DIR}
    wget --show-progress -nc -q https://zenodo.org/records/2546921/files/${FILE}?download=1 -O ${OUTPUT_FILE}
    yes n | gzip -d ${OUTPUT_FILE}
done
