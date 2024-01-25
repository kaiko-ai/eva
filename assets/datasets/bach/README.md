# BACH

The BACH dataset consists of 408 labelled patches from 4 classes ("Normal", "Benign", "Invasive", "InSitu"). It was used for the "BACH Grand Challenge on Breast Cancer Histology images".


## Key stats raw data

|   |   |
| --- | --- |
| Modality | Vision (patches from WSIs) |
| Type of task | Multiclass classification (4 classes) |
| Type of cancer | Breast |
| Data size | total: 10.4GB / data in use: 7.37 GB (18.9 MB per image) |
| Number of files in use | 408 (102 from each class)|
| Files format | `.tif` images|
| Splits in use | one labelled split |


## Directory structure raw data

The data `ICIAR2018_BACH_Challenge.zip` from [zenodo](https://zenodo.org/records/3632035) is organized as follows:

```
ICAR2018_BACH_Challenge
├── Photos                    # All labelled patches used by eva
│   ├── Normal
│   │   ├── n032.tif
│   │   └── ...
│   ├── Benign
│   │   └── ...
│   ├── Invasive
│   │   └── ...
│   ├── InSitu
│   │   └── ...
├── WSI                       # WSIs, not in use
│   ├── ...
└── ...
```

## Data preprocessing

The BACH data will be downloaded and preprocessed by the `BachDataset` class
|   |   |
| --- | --- |
| Data download | class method using [link to download](https://zenodo.org/records/3632035/files/ICIAR2018_BACH_Challenge.zip?download=1) |
| Labels | Derived directly from the folder structure. |
| Splits method | images ordered by filename, stratfied by label |
| Splits ratio | train: 70%, val: 15%, test: 15% |

## Relevant links

* [BACH dataset on zenodo](https://zenodo.org/records/3632035)
* [Direct link to download](https://zenodo.org/records/3632035/files/ICIAR2018_BACH_Challenge.zip?download=1)
* [BACH challenge website](https://iciar2018-challenge.grand-challenge.org/home/)


## License

[Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode)
