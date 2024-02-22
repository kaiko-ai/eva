# CRC-HE

The CRC-HE dataset consists of labelled patches (9 classes) from colorectal cancer (CRC) and normal tissue. We use the `NCT-CRC-HE-100K` dataset for training and validation and the `CRC-VAL-HE-7K for testing`. 

The `NCT-CRC-HE-100K-NONORM` consists of 100,000 images without applied color normalization. The `CRC-VAL-HE-7K` consists of 7,180 image patches from 50 patients without overlap with `NCT-CRC-HE-100K-NONORM`.

The tissue classes are: Adipose (ADI), background (BACK), debris (DEB), lymphocytes (LYM), mucus (MUC), smooth muscle (MUS), normal colon mucosa (NORM), cancer-associated stroma (STR) and colorectal adenocarcinoma epithelium (TUM)

## Raw data

### Key stats

|                      |                                                          |
|----------------------|----------------------------------------------------------|
| **Modality**         | Vision (WSI patches)                                     |
| **Task**             | Multiclass classification (9 classes)                    |
| **Cancer type**      | Colorectal                                               |
| **Data size**        | total: 11.7GB (train/val), 800MB (test)                  |
| **Image dimension**  | 224 x 224 x 3                                            |
| **FoV (μm/px)**      | 20x (0.5)                                                |
| **Files format**     | `.tif` images                                            |
| **Number of images** | 107,180 (100k train/val, 7.2k test)                      |
| **Splits in use**    | NCT-CRC-HE-100K-NONORM (train/val), CRC-VAL-HE-7K (test) |


### Organization

The data `NCT-CRC-HE-100K-NONORM.zip` and `CRC-VAL-HE-7K.zip` from [zenodo](https://zenodo.org/records/1214456) are organized as follows:

```
NCT-CRC-HE-100K-NONORM         # All images used for training and validation
├── ADI                        # All labelled patches belonging to the 1st class
│   ├── ADI-AAAFLCLY.tif
│   ├── ...
├── BACK                       # All labelled patches belonging to the 2nd class
│   ├── ...
└── ...

CRC-VAL-HE-7K                  # All images used for testing
├── ...                        # identical structure as for NCT-CRC-HE-100K-NONORM
└── ...
```

## Download and preprocessing

The `CRC_HE` dataset class supports download the data no runtime with the initialized argument
`download: bool = True`.

The splits are created from the indices specified in the `CRC_HE` dataset class. The indices were selected to ensure a
80% / 20% ordered and stratified train/val split from the `NCT-CRC-HE-100K-NONORM` dataset. The test split is the complete
`CRC-VAL-HE-7K` dataset.

| Splits | Train          | Validation     | Validation   | 
|---|----------------|----------------|--------------|
| #Samples | 80,003 (74.6%) | 19,997 (18.7%) | 7,180 (6.7%) |


## Relevant links

* [CRC-HE datasets on zenodo](https://zenodo.org/records/1214456)
* [Reference API Vision dataset classes](../reference/vision/data/datasets.md)


## License

[CC BY 4.0 LEGAL CODE](https://creativecommons.org/licenses/by/4.0/legalcode)
