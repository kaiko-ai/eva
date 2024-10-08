# BCSS

The BCSS (Breast Cancer Semantic Segmentation) consists of extracts from 151 WSI images from [TCGA](https://www.cancer.gov/ccg/research/genome-sequencing/tcga), containing over 20,000 segmentation annotations covering 21 different tissue types.


## Raw data

### Key stats

|                       |                                                           |
|-----------------------|-----------------------------------------------------------|
| **Modality**          | Vision (WSI extracts)                                      |
| **Task**              | Segmentation - 22 classes (tissue types)|
| **Data size**         | total: ~5GB                                             |
| **Image dimension**   | ~1000-3000 x ~1000-3000 x 3                                           |
| **Magnification (μm/px)**  | 40x (0.25)                                       |
| **Files format**      | `.png` images / `.mat` segmentation masks                 |
| **Number of images**  | 151                                                        |
| **Splits in use**     | Train, Val and Test                                            |


### Organization

The data is organized as follows:

```
bcss
├── rgbs_colorNormalized       # wsi images
│   ├── TCGA-*.png
├── masks                      # segmentation masks
│   ├── TCGA-*.png             # same filenames as images 
```

## Download and preprocessing

The `BCSS` dataset class doesn't download the data during runtime and must be downloaded manually from links provided [here](https://drive.google.com/drive/folders/1zqbdkQF8i5cEmZOGmbdQm-EP8dRYtvss?usp=sharing).

Although the original images have a resolution of 0.25 microns per pixel (mpp), we extract patches at 0.5 mpp for evaluation. This is because using the original resolution with common foundation model patch sizes (e.g. 224x224 pixels) would result in regions that are too small, leading to less expressive segmentation masks and unnecessarily complicating the task.


### Splits

As a test set, we use the images from the medical institues OL, LL, E2, EW, GM, and S3, as proposed by the [authors](https://bcsegmentation.grand-challenge.org/Baseline/). For the validation split, we use images from the institutes BH, C8, A8, A1 and E9, which results in the following dataset sizes:


| Splits   | Train       | Validation  | Test       |  
|----------|-------------|-------------|------------|
| #Samples | 76 (50.3%)  | 30 (19.9%)  | 45 (29.8%)|


## Relevant links

* [Dataset Repo](https://github.com/PathologyDataScience/BCSS)
* [Breast Cancer Segmentation Grand Challenge](https://bcsegmentation.grand-challenge.org)
* [Google Drive Download Link for 0.25 mpp version](https://drive.google.com/drive/folders/1zqbdkQF8i5cEmZOGmbdQm-EP8dRYtvss?usp=sharing)

## License

The BCSS dataset is held under the [CC0 1.0 UNIVERSAL](https://creativecommons.org/publicdomain/zero/1.0/) license.
