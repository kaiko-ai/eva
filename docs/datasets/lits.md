# LiTS17 (Liver Tumor Segmentation Challenge 2017)

LiTS17 is a liver tumor segmentation benchmark. The data and segmentations are provided by various clinical sites around the world. The training data set contains 130 CT scans and the test data set 70 CT scans.

The segmentation classes are: Background, Liver and Tumor.

## Raw data

### Key stats

|                       |                                                           |
|-----------------------|-----------------------------------------------------------|
| **Modality**          | Vision (radiology, CT scans)                              |
| **Task**              | Segmentation (3 classes)                                  |
| **Data size**         | train: 15GB (53.66 GB uncompressed)                       |
| **Image dimension**   | ~300 x ~300 x ~350 (number of slices) x 1 (grey scale) *  |
| **Files format**      | `.nii` ("NIFTI") images                                   |
| **Number of scans**   | 131 (58638 slices)                                        |
| **Splits in use**     | train (70%) / val (15%) / test (15%)                  |


### Splits

We use the following random split:

| Splits         | Train            | Validation        | Test             |
|----------------|------------------|-------------------|------------------|
| #Scans; Slices | 91; 38686 (77%) | 19; 11192 (11.5%) | 21; 8760 (11.5%) |


### Organization

The training data are organized as follows:

```
Training Batch 1               # Train images part 1
├── segmentation-0.nii         # Semantic labels for volume 0
├── segmentation-1.nii         # Semantic labels for volume 1
├── ...
├── volume-0.nii               # CT-Scan 0
├── volume-1.nii               # CT-Scan 1
└── ...

Training Batch 2               # Train images part 2
├── segmentation-28.nii        # Semantic labels for volume 28
├── segmentation-29.nii        # Semantic labels for volume 29
├── ...
├── volume-28.nii              # CT-Scan 28
├── volume-29.nii              # CT-Scan 29
└── ...
```

## Download and preprocessing

The `LiTS` dataset can be downloaded from the official
[LiTS competition page](https://competitions.codalab.org/competitions/17094).
The training split comes into two `.zip` files, namely `Training_Batch1.zip`
and `Training_Batch2.zip`, which should be extracted and merged.

## Relevant links

* [LiTS - Liver Tumor Segmentation Challenge](https://competitions.codalab.org/competitions/17094)
* [Whitepaper](https://arxiv.org/pdf/1901.04056)


## License

[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en)
