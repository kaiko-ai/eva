# PANDA (Prostate cANcer graDe Assessment)

The PANDA datasets consists of 10616 whole-slide images of digitized H&E-stained prostate tissue biopsies originating from two medical centers. After the biopsy, the slides were classified into Gleason patterns (3, 4 or 5) based on the architectural growth patterns of the tumor, which are then converted into an ISUP grade on a 0-5 scale.

The Gleason grading system is the most important prognostic marker for prostate cancer and the ISUP grade has a crucial role when deciding how a patient should be treated. However, the system suffers from significant inter-observer variability between pathologists, leading to imperfect and noisy labels.

Source: https://www.kaggle.com/competitions/prostate-cancer-grade-assessment


## Raw data

### Key stats

|                           |                                                          |
|---------------------------|----------------------------------------------------------|
| **Modality**              | Vision (WSI)                                             |
| **Task**                  | Multiclass classification (6 classes)                    |
| **Cancer type**           | Prostate                                                 |
| **Data size**             | 347 GB                                                   |
| **Image dimension**       | ~20k x 20k x 3                                           |
| **Magnification (μm/px)** | 20x (0.5) - Level 0                                      |
| **Files format**          | `.tiff`                                                  |
| **Number of images**      | 10616 (9555 after removing noisy labels)                 |


### Organization

The data `prostate-cancer-grade-assessment.zip` from [kaggle](https://www.kaggle.com/competitions/prostate-cancer-grade-assessment/data) is organized as follows:

```
prostate-cancer-grade-assessment
├── train_images
│   ├── 0005f7aaab2800f6170c399693a96917.tiff
│   └── ...
├── train_label_masks (not used in eva)
│   ├── 0005f7aaab2800f6170c399693a96917_mask.tiff
│   └── ...
├── train.csv (contains Gleason & ISUP labels)
├── test.csv
├── sample_submission.csv
```

## Download and preprocessing

The `PANDA` dataset class doesn't download the data during runtime and must be downloaded manually from [kaggle](https://www.kaggle.com/competitions/prostate-cancer-grade-assessment/data).

As done in other studies<sup>1</sup> we exclude ~10% of the samples with noisy labels according to kaggle's [6th place solution](https://www.kaggle.com/competitions/prostate-cancer-grade-assessment/discussion/169230) resulting in a total dataset size of 9555 WSIs.

We then generate random stratified train / validation and test splits using a 0.7 / 0.15 / 0.15 ratio:


| Splits   | Train       | Validation  | Test       |  
|----------|-------------|-------------|------------|
| #Samples | 6686 (70%)  | 1430 (15%)  | 1439 (15%) |


## Relevant links

* [Kaggle Challenge](https://www.kaggle.com/competitions/prostate-cancer-grade-assessment)
* [Noisy Labels](https://github.com/analokmaus/kaggle-panda-challenge-public)


## License

[CC BY-SA-NC 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)

## References
1 : [A General-Purpose Self-Supervised Model for Computational Pathology](https://arxiv.org/abs/2308.15474)