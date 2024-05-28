# Camelyon16

The Camelyon16 dataset consists of 400 WSIs of lymph nodes for breast cancer metastasis classification. The dataset is a combination of data from two independent datasets, collected from two separate medical centers in the Netherlands (Radboud University Medical Center and University Medical Center Utrecht). The dataset contains the slides from which [PatchCamelyon](patch_camelyon.md)-patches were extracted.

The dataset is divided in a train set (270 slides) and test set (130 slides), both containing images from both centers.

The task was part of [Grand Challenge](https://grand-challenge.org/) in 2016 and has later been replaced by Camelyon17.

Source: https://camelyon16.grand-challenge.org

## Raw data

### Key stats

|                           |                                                          |
|---------------------------|----------------------------------------------------------|
| **Modality**              | Vision (Slide-level)                                     |
| **Task**                  | Binary classification                                    |
| **Cancer type**           | Breast                                                   |
| **Data size**             | ~700 GB                                                  |
| **Image dimension**       | ~100-250k x ~100-250k x 3                                |
| **Magnification (μm/px)** | 40x (0.25) - Level 0                                     |
| **Files format**          | `.tif`                                                   |
| **Number of images**      | 400 (270 train, 130 test)                                |


### Organization

The data `CAMELYON16` (download links [here](https://camelyon17.grand-challenge.org/Data/)) is organized as follows:

```
CAMELYON16
├── training
│   ├── normal
|   │   ├── normal_001.tif
|   │   └── ...
│   ├── tumor
|   │   ├── tumor_001.tif
|   │   └── ...
│   └── lesion_annotations.zip
├── testing
│   ├── images
|   │   ├── test_001.tif
|   │   └── ...
│   ├── evaluation     # masks not in use
│   ├── reference.csv  # targets
│   └── lesion_annotations.zip
```

## Download and preprocessing

The `Camelyon16` dataset class doesn't download the data during runtime and must be downloaded manually from links provided [here](https://camelyon17.grand-challenge.org/Data/).

The dataset is split into train / test. As in [1], we additionally split the train set into a stratified 90:10 train/val split.

| Splits   | Train       | Validation  | Test       |  
|----------|-------------|-------------|------------|
| #Samples | 243 (60.75%)  | 27 (6.75%)  | 130 (32.5%) |


## Relevant links

* [Grand Challenge dataset description](https://camelyon16.grand-challenge.org/Data/)
* [Download links](https://camelyon17.grand-challenge.org/Data/)


## References
1 : [A General-Purpose Self-Supervised Model for Computational Pathology](https://arxiv.org/abs/2308.15474)