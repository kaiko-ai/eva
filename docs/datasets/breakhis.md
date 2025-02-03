# BreakHis

The Breast Cancer Histopathological Image Classification (BreakHis) is  composed of 9,109 microscopic images of breast tumor tissue collected from 82 patients using different magnifying factors (40X, 100X, 200X, and 400X). To date, it contains 2,480 benign and 5,429 malignant samples (700X460 pixels, 3-channel RGB, 8-bit depth in each channel, PNG format). This database has been built in collaboration with the P&D Laboratory, Pathological Anatomy and Cytopathology, Parana, Brazil.

The dataset is divided into two main groups: benign tumors and malignant tumors. The dataset currently contains four histological distinct types of benign breast tumors: adenosis (A), fibroadenoma (F), phyllodes tumor (PT), and tubular adenona (TA); and four malignant tumors (breast cancer): carcinoma (DC), lobular carcinoma (LC), mucinous carcinoma (MC) and papillary carcinoma (PC).

## Raw data

### Key stats

|                                |                             |
|--------------------------------|-----------------------------|
| **Modality**                   | Vision (WSI patches)        |
| **Task**                       | Multiclass classification (6 classes) |
| **Cancer type**                | Breast                      |
| **Data size**                  | 4 GB                        |
| **Image dimension**            | 1812 x 1812                 |
| **Magnification (μm/px)**      | 40x (0.25)                  |
| **Files format**               | `png`                       |
| **Number of images**           | 1995                        |


### Splits

The data source provides train/validation/test splits

| Splits | Train           | Validation   |
|----------|---------------|--------------|
| #Samples | 1393 (70%)    | 602 (30%)    |


### Organization

The BreakHis data is organized as follows:

```
BreaKHis_v1
├── histology_slides
│   ├── breast
|   │   ├── benign
|   │   |   ├── SOB
|   │   |   |   ├── adenosis
|   │   |   |   ├── fibroadenoma
|   │   |   |   └── ...
```


## Download and preprocessing
The `UniToPatho` dataset class doesn't download the data during runtime and must be downloaded manually from [the official source](https://ieee-dataport.org/open-access/unitopatho).

## Relevant links

* [Official Source](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)"