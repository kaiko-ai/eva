# Gleason (Arvaniti)

Benchmark dataset for automated Gleason grading of prostate cancer tissue microarrays via deep learning as proposed by [Arvaniti et al.](https://www.nature.com/articles/s41598-018-30535-1).

Classify image patches as benign, Gleason pattern 3, 4 or 5. For the test dataset, we use the labels from pathologist Nr. 1.

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
| **Number of images**           | TODO                        |


### Splits

We use the same splits as proposed in the paper:.

| Splits | Train         | Validation   | Test         |
|---|---------------|--------------|--------------|
| #Samples | 262,144 (80%) | 32,768 (10%) | 32,768 (10%) |

Note that the authors chose TMA 76 as validation cohort because it contains the most balanced distribution of Gleason scores.


## Download and preprocessing
The `GleasonArvaniti` dataset class doesn't download the data during runtime and must be downloaded and preprocessed manually:

1. Download dataset archives from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OCYCMP
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

