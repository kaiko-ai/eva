# Beyond the Cranial Vault (BTCV) Abdomen dataset.

The BTCV dataset comprises abdominal CT scans acquired at the Vanderbilt University Medical Center from metastatic liver cancer patients or post-operative ventral hernia patients. 

The annotations cover segmentations of the spleen, right and left kidney, gallbladder, esophagus, liver, stomach, aorta, inferior vena cava, portal vein and splenic vein, pancreas, right adrenal gland, left adrenal gland are included in this data set. Images were manually labeled by two experienced undergraduate students, and verified by a radiologist.


## Raw data

### Key stats

|                       |                                                           |
|-----------------------|-----------------------------------------------------------|
| **Modality**          | Vision (radiology, CT scans)                              |
| **Task**              | Segmentation (14 classes)                                  |
| **Image dimension**   | 512 x 512 x ~140 (number of slices)  |
| **Files format**      | `.nii` ("NIFTI") images                                   |
| **Number of scans**   | 30                                        |
| **Splits in use**     | train (80%) / val (20%)                  |


### Splits

While the full dataset contains 90 CT scans, we use the train/val split from MONAI which uses a subset of 30 CT scans (https://github.com/Luffy03/Large-Scale-Medical/blob/main/Downstream/monai/BTCV/dataset/dataset_0.json):

| Splits         | Train            | Validation        |
|----------------|------------------|-------------------|
| #Scans | 24 (80%) | 6 (20%) |


### Organization

The training data are organized as follows:

```
imagesTr
├── img0001.nii.gz
├── img0002.nii.gz 
└── ...

labelsTr
├── label0001.nii.gz
├── label0002.nii.gz
└── ...
```

## Download

The `BTCV` dataset class supports downloading the data during runtime by setting the init argument `download=True`.

## Relevant links

* [zenodo download source](https://zenodo.org/records/1169361)
* [huggingface dataset](https://huggingface.co/datasets/Luffy503/VoCo_Downstream/blob/main/BTCV.zip)


## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0)
