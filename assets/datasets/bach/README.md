# BACH

The BACH dataset consists of microscopy and WSI images, of which we use only the microscopy images. These are 408 labelled images from 4 classes ("Normal", "Benign", "Invasive", "InSitu"). This dataset was used for the "BACH Grand Challenge on Breast Cancer Histology images".


## Raw data

### Key stats

|                      |                                                         |
|----------------------|---------------------------------------------------------|
| **Modality**         | Vision (microscopy images)                             |
| **Task**             | Multiclass classification (4 classes)                   |
| **Cancer type**      | Breast                                                  |
| **Data size**        | total: 10.4GB / data in use: 7.37 GB (18.9 MB per image) |
| **Image dimension**  | 1536 x 2048 x 3                                         |
| **Files format**     | `.tif` images                                           |
| **Number of images** | 408 (102 from each class)                               |
| **Splits in use**    | one labelled split                                      |


### Organization

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

## Download and preprocessing

The dataset class `Bach` supports download the data no runtime with the initialized argument
`download: bool = True`.

The splits are created by ordering images by filename and stratifying by label

| Splits | Train        | Validation | Test         |
|---|--------------|------------|--------------|
| #Samples | 286 (70%) | 61 (15%)   | 61 (15%) |


## Relevant links

* [BACH dataset on zenodo](https://zenodo.org/records/3632035)
* [Direct link to download](https://zenodo.org/records/3632035/files/ICIAR2018_BACH_Challenge.zip?download=1)
* [BACH challenge website](https://iciar2018-challenge.grand-challenge.org/home/)


## License

[Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode)
