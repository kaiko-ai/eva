# PatchCamelyon


The PatchCamelyon benchmark is a image classification dataset with 327,680 color images (96 x 96px) extracted from histopathologic scans of lymph node sections. Each image is annotated with a binary label indicating presence of metastatic tissue.

## Raw data

### Key stats

|                      |                             |
|----------------------|-----------------------------|
| **Modality**         | Vision (WSI patches)        |
| **Task**             | Binary classification       |
| **Cancer type**      | Breast                      |
| **Data size**        | 8 GB                        |
| **Image dimension**  | 96 x 96 x 3                 |
| **FoV (μm/px)**      | 10x (1.0) \*                |
| **Files format**     | `h5`                        |
| **Number of images** | 327,680 (50% of each class) |


\* The slides were acquired and digitized at 2 different centres using a 40x objective but under-sampled to 10x to increase the field of view. Some papers do categorize it as 10x. Basically artificial 10x patches.

### Splits

The datasource provides train/validation/test splits

| Splits | Train         | Validation   | Test         |
|---|---------------|--------------|--------------|
| #Samples | 262,144 (80%) | 32,768 (10%) | 32,768 (10%) |


### Organization

The PatchCamelyon data from [zenodo](https://zenodo.org/records/2546921) is organized as follows:

```
├── camelyonpatch_level_2_split_train_x.h5.gz               # train images
├── camelyonpatch_level_2_split_train_y.h5.gz               # train labels
├── camelyonpatch_level_2_split_valid_x.h5.gz               # val images
├── camelyonpatch_level_2_split_valid_y.h5.gz               # val labels
├── camelyonpatch_level_2_split_test_x.h5.gz                # test images
├── camelyonpatch_level_2_split_test_y.h5.gz                # test labels
```


## Download and preprocessing
The dataset class `PatchCamelyon` supports download the data no runtime with the initialized argument
`download: bool = True`.

Labels are provided by source files, splits are given by file names.

## Relevant links

* [PatchCamelyon dataset on zenodo](https://zenodo.org/records/2546921)
* [GitHub repository](https://github.com/basveeling/pcam)

## Citation
```
@misc{b_s_veeling_j_linmans_j_winkens_t_cohen_2018_2546921,
  author       = {B. S. Veeling, J. Linmans, J. Winkens, T. Cohen, M. Welling},
  title        = {Rotation Equivariant CNNs for Digital Pathology},
  month        = sep,
  year         = 2018,
  doi          = {10.1007/978-3-030-00934-2_24},
  url          = {https://doi.org/10.1007/978-3-030-00934-2_24}
}
```

## License

[Creative Commons Zero v1.0 Universal](https://choosealicense.com/licenses/cc0-1.0/)
