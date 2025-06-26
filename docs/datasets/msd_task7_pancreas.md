# MSD Task 7 Pancreas

MSD Task 7 Pancreas is part of the Medical Segmentation Decathlon (MSD) challenge. The dataset consists of 420 portal-venous phase CT scans of patients undergoing resection of pancreatic masses. The corresponding target ROIs were the pancreatic parenchyma and pancreatic mass (cyst or tumor). This dataset was selected due to label unbalance between large (background), medium (pancreas) and small (tumor) structures. The data was acquired in the Memorial Sloan Kettering Cancer Center, New York, US.

The segmentation classes are: Background, Pancreas and Cancer.

## Raw data

### Key stats

|                       |                                                           |
|-----------------------|-----------------------------------------------------------|
| **Modality**          | Vision (radiology, CT scans)                              |
| **Task**              | Segmentation (3 classes)                                  |
| **Data size**         | 11 GB                                                     |
| **Image dimension**   | Variable (3D volumes)                                     |
| **Files format**      | `.nii.gz` ("NIFTI") images                                |
| **Number of scans**   | 281                                                       |
| **Splits in use**     | train / val                                               |


### Splits

The dataset uses predefined train/validation splits:

| Splits         | Train            | Validation        |
|----------------|------------------|-------------------|
| # Scans        | 257              | 24                |

The split was taken from https://github.com/Luffy03/Large-Scale-Medical/blob/main/Downstream/monai/Panc/dataset_panc.json

### Organization

The training data is expected to be organized as follows:

```
Dataset007_Pancreas
├── imagesTr/
│   ├── pancreas_001_0000.nii.gz
│   ├── pancreas_002_0000.nii.gz
│   └── ...
└── labelsTr/
    ├── pancreas_001.nii.gz
    ├── pancreas_002.nii.gz
    └── ...
```

## Download and preprocessing

The `MSDTask7Pancreas` dataset can be downloaded automatically by setting `download=True` when initializing the dataset, or by setting the environment variable `DOWNLOAD_DATA=true`. The dataset is hosted on Hugging Face and requires a Hugging Face token to be set in the `HF_TOKEN` environment variable.

## Relevant links

* [Medical Segmentation Decathlon Paper](https://www.nature.com/articles/s41467-022-30695-9)
* [Dataset source on Hugging Face](https://huggingface.co/datasets/Luffy503/VoCo_Downstream)

## License

Please refer to the original dataset license terms from the Medical Segmentation Decathlon.