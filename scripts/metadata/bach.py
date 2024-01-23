"""script to create metadata for the BACH dataset

This script creates a metadata file with paths, labels and splits of the 
BACH train patch dataset. These are 400 patches in  
"ICIAR2018_BACH_Challenge/Photos" that belong to one of the four classes: 
"Normal", "Benign", "InSitu", "Invasive".

The metadata is a dataframe, that will be stored as csv with the columns:
    - path: path to the raw images
    - label: label of the image
    - split: split the image belongs to ('train', 'val', 'test')*

* the splits are created as follows: ordered by filename, stratfied by label,
  train: 70%, val: 15%, test: 15%

to run:
(1) download 'ICIAR2018_BACH_Challenge.zip' from https://zenodo.org/records/3632035
    and unzip the data (only the 'Photos' folder is needed)
(2) specify the following parameters:
    - DOWNLOADED_DATA_PATH: the directory where the raw patches are stored, e.g.
      "<...>/3632035/ICIAR2018_BACH_Challenge/Photos"
    - TARGET_METADATA_FILE: the directory where the metadata dataframe will be 
      stored (e.g. "./metadata").
    - TARGET_METADATA_FILE: the metadata file name (e.g. "bach_metadata.csv")
(3) run script with `python scripts/metadata_creation/bach.py`
"""

import glob
import os

import pandas as pd
from loguru import logger

# Specify relevant directories:
DOWNLOADED_DATA_PATH = "<...>/3632035/ICIAR2018_BACH_Challenge/Photos"
TARGET_METADATA_DIR = "./metadata"
TARGET_METADATA_FILE = "bach_metadata.csv"

# labels mapping and train/val/test split fractions (do not modify):
_file_dir_to_label = {
    "Normal": 0,
    "Benign": 1,
    "InSitu": 2,
    "Invasive": 3,
}
_train_fraction = 0.7
_val_fraction = 0.15


def main():
    # create dataframe with paths and labels:
    all_patches = glob.glob(f"{DOWNLOADED_DATA_PATH}/**/*.tif")
    logger.info(f"Loaded paths to {len(all_patches)} images.")

    df_metadata = pd.DataFrame(all_patches, columns=["path"])
    df_metadata["label"] = df_metadata["path"].apply(lambda x: _file_dir_to_label[x.split("/")[-2]])

    # create splits:
    df_metadata["split"] = ""
    dfs_label = []
    for label in df_metadata["label"].unique():
        df = (
            df_metadata[df_metadata["label"] == label].sort_values(by="path").reset_index(drop=True)
        )
        n_train = round(df.shape[0] * _train_fraction)
        n_val = round(df.shape[0] * _val_fraction)
        df.loc[:n_train, "split"] = "train"
        df.loc[n_train : n_train + n_val, "split"] = "val"
        df.loc[n_train + n_val :, "split"] = "test"
        dfs_label.append(df)
    df_metadata = pd.concat(dfs_label).sort_values(by=["split", "label"]).reset_index(drop=True)

    # save metadata:
    if not os.path.exists(TARGET_METADATA_DIR):
        os.mkdir(TARGET_METADATA_DIR)
    df_metadata.to_csv(os.path.join(TARGET_METADATA_DIR, TARGET_METADATA_FILE), index=False)
    logger.info(f"Metadata saved to {os.path.join(TARGET_METADATA_DIR, TARGET_METADATA_FILE)}.")


if __name__ == "__main__":
    main()
