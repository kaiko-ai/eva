# Gleason (Arvaniti)

Benchmark dataset for automated Gleason grading of prostate cancer tissue microarrays via deep learning as proposed by [Arvaniti et al.](https://www.nature.com/articles/s41598-018-30535-1).

Images are classified as benign, Gleason pattern 3, 4 or 5. The dataset contains annotations on a discovery / train cohort of 641 patients and an independent test cohort of 245 patients annotated by two pathologists. For the test cohort, we only use the labels from pathologist Nr. 1 for this benchmark

## Raw data

### Key stats

|                                |                             |
|--------------------------------|-----------------------------|
| **Modality**                   | Vision (WSI patches)        |
| **Task**                       | Multiclass classification (4 classes) |
| **Cancer type**                | Prostate                    |
| **Data size**                  | 4 GB                        |
| **Image dimension**            | 750 x 750                   |
| **Magnification (μm/px)**      | 40x (0.23)                  |
| **Files format**               | `jpg`                       |
| **Number of images**           | 22,752                       |


### Splits

The following splits are proposed in the paper:

| Splits   | Train           | Validation     | Test           |
|----------|-----------------|----------------|----------------|
| #Samples | 15,303 (67.26%) | 2,482 (10.91%) | 4,967 (21.83%) |

Note that the authors chose TMA 76 as validation cohort because it contains the most balanced distribution of Gleason scores.
We couldn't achieve stable results when evaluating on the test set, so we only use the train and validation sets for this benchmark.

## Download and preprocessing
The `GleasonArvaniti` dataset class doesn't download the data during runtime and must be downloaded and preprocessed manually:

1. Download dataset archives from the [official source](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OCYCMP)
2. Unpack all .tar.gz archives into the same folder
3. Adjust the folder structure and then run the `create_patches.py` from https://github.com/eiriniar/gleason_CNN/tree/master

This should result in the folloing folder structure:

```
arvaniti_gleason_patches
├── test_patches_750
│   ├── patho_1
│   │   ├── ZT80_38_A_1_1
    │   │   ├── ZT76_39_A_1_1_patch_12_class_0.jpg
    │   │   ├── ZT76_39_A_1_1_patch_23_class_0.jpg
│   │   │   └── ...
│   │   ├── ZT80_38_A_1_2
│   │   │   └── ...
│   │   └── ...
│   ├── patho_2  # we don't use this
│   │   └── ...
├── train_validation_patches_750
│   ├── ZT76_39_A_1_1
│   │   ├── ZT76_39_A_1_1_patch_12_class_0.jpg
│   │   ├── ZT76_39_A_1_1_patch_23_class_0.jpg
│   │   └── ...
│   ├── ZT76_39_A_1_2
│   └── ...
```

## Relevant links

* [Paper](https://www.nature.com/articles/s41598-018-30535-1)
* [GitHub](https://github.com/eiriniar/gleason_CNN)
* [Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OCYCMP)

## License

[CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)

