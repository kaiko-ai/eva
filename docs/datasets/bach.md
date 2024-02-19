# BACH

The BACH dataset consists of microscopy and WSI images, of which we use only the microscopy images. These are 408 labelled images from 4 classes ("Normal", "Benign", "Invasive", "InSitu"). This dataset was used for the "BACH Grand Challenge on Breast Cancer Histology images".


## Raw data

### Key stats

|                      |                                                          |
|----------------------|----------------------------------------------------------|
| **Modality**         | Vision (microscopy images)                               |
| **Task**             | Multiclass classification (4 classes)                    |
| **Cancer type**      | Breast                                                   |
| **Data size**        | total: 10.4GB / data in use: 7.37 GB (18.9 MB per image) |
| **Image dimension**  | 1536 x 2048 x 3                                          |
| **FoV (μm/px)**      | 20x (0.5)                                                |
| **Files format**     | `.tif` images                                            |
| **Number of images** | 408 (102 from each class)                                |
| **Splits in use**    | one labelled split                                       |


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

The `BACH` dataset class supports download the data no runtime with the initialized argument
`download: bool = True`.

The splits are created from the indices specified in te BACH dataset class. These indices were picked to prevent data 
leakage due to images belonging to the same patient. Because the small dataset in combination with the patient ID constraint 
does not allow to split the data three-ways with sufficient amount of data in each split, we only create a train and val 
split and leave it to the user to submit predictions on the official test split to the [BACH Challenge Leaderboard](https://iciar2018-challenge.grand-challenge.org/evaluation/challenge/leaderboard/).

| Splits | Train     | Validation | 
|---|-----------|------------|
| #Samples | 268 (67%) | 132 (33%)  |


## Relevant links

* [BACH dataset on zenodo](https://zenodo.org/records/3632035)
* [BACH Challenge website](https://iciar2018-challenge.grand-challenge.org/)
* [BACH Challenge Leaderboard](https://iciar2018-challenge.grand-challenge.org/evaluation/challenge/leaderboard/)
* [Patient ID information](https://www.dropbox.com/sh/sc7yg21bcs3wr0z/AACiavY0BQPF6GYna9Fkjzola?e=1&dl=0) (Link provided on BACH challenge website)
* [Reference API BACH dataset class](../reference/vision/data/bach.md)


## License

[Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode)
