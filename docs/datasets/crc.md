# CRC

The CRC-HE dataset consists of labelled patches (9 classes) from colorectal cancer (CRC) and normal tissue. We use the `NCT-CRC-HE-100K` dataset for training and validation and the `CRC-VAL-HE-7K for testing`. 

The `NCT-CRC-HE-100K-NONORM` consists of 100,000 images without applied color normalization. The `CRC-VAL-HE-7K` consists of 7,180 image patches from 50 patients without overlap with `NCT-CRC-HE-100K-NONORM`.

The tissue classes are: Adipose (ADI), background (BACK), debris (DEB), lymphocytes (LYM), mucus (MUC), smooth muscle (MUS), normal colon mucosa (NORM), cancer-associated stroma (STR) and colorectal adenocarcinoma epithelium (TUM)

## Raw data

### Key stats

|                                |                                                     |
|--------------------------------|-----------------------------------------------------|
| **Modality**                   | Vision (WSI patches)                                |
| **Task**                       | Multiclass classification (9 classes)               |
| **Cancer type**                | Colorectal                                          |
| **Data size**                  | total: 11.7GB (train), 800MB (val)                  |
| **Image dimension**            | 224 x 224 x 3                                       |
| **Magnification (μm/px)**      | 20x (0.5)                                           |
| **Files format**               | `.tif` images                                       |
| **Number of images**           | 107,180 (100k train, 7.2k val)                      |
| **Splits in use**              | NCT-CRC-HE-100K (train), CRC-VAL-HE-7K (val)        |


### Splits

We use the splits according to the data sources:

 - Train split: `NCT-CRC-HE-100K`
 - Validation split: `CRC-VAL-HE-7K`

| Splits   | Train           | Validation   | 
|----------|-----------------|--------------|
| #Samples | 100,000 (93.3%) | 7,180 (6.7%) | 

A test split is not provided. Because the patient information for the training data is not available, dividing the 
training data in a train/val split (and using the given val split as test split) is not possible without risking data leakage.
__eva__ therefore reports evaluation results for CRC HE on the validation split.

### Organization

The data `NCT-CRC-HE-100K.zip`, `NCT-CRC-HE-100K-NONORM.zip` and `CRC-VAL-HE-7K.zip`
from [zenodo](https://zenodo.org/records/1214456) are organized as follows:

```
NCT-CRC-HE-100K                # All images used for training
├── ADI                        # All labelled patches belonging to the 1st class
│   ├── ADI-AAAFLCLY.tif
│   ├── ...
├── BACK                       # All labelled patches belonging to the 2nd class
│   ├── ...
└── ...

NCT-CRC-HE-100K-NONORM         # All images used for training
├── ADI                        # All labelled patches belonging to the 1st class
│   ├── ADI-AAAFLCLY.tif
│   ├── ...
├── BACK                       # All labelled patches belonging to the 2nd class
│   ├── ...
└── ...

CRC-VAL-HE-7K                  # All images used for validation
├── ...                        # identical structure as for NCT-CRC-HE-100K-NONORM
└── ...
```

## Download and preprocessing

The `CRC` dataset class supports download the data no runtime with the initialized argument
`download: bool = True`.

## Relevant links

* [CRC datasets on zenodo](https://zenodo.org/records/1214456)
* [Reference API Vision dataset classes](../reference/vision/data/datasets.md)


## License

[CC BY 4.0 LEGAL CODE](https://creativecommons.org/licenses/by/4.0/legalcode)
