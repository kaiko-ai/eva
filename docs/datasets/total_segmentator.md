# TotalSegmentator

The TotalSegmentator dataset is an radiology image-segmentation dataset with 1228 3D images and corresponding masks with 117 different anatomical structures. It can be used for segmentation and multilabel classification tasks.

## Raw data

### Key stats

|                       |                                                           |
|-----------------------|-----------------------------------------------------------|
| **Modality**          | Vision (radiology, CT scans)                              |
| **Task**              | Segmentation / multilabel classification (117 classes)    |
| **Data size**         | total: 23.6GB                                             |
| **Image dimension**   | ~300 x ~300 x ~350 (number of slices) x 1 (grey scale) *  |
| **Files format**      | `.nii` ("NIFTI") images                                   |
| **Number of images**  | 1228                                                      |
| **Splits in use**     | one labelled split                                        |

/* image resolution and number of slices per image vary

### Organization

The data `Totalsegmentator_dataset_v201.zip` from [zenodo](https://zenodo.org/records/10047292) is organized as follows:

```
Totalsegmentator_dataset_v201
├── s0011                               # one image
│   ├── ct.nii.gz                       # CT scan
│   ├── segmentations                   # directory with segmentation masks
│   │   ├── adrenal_gland_left.nii.gz   # segmentation mask 1st anatomical structure
│   │   ├── adrenal_gland_right.nii.gz  # segmentation mask 2nd anatomical structure
│   │   └── ...
└── ...
```

## Download and preprocessing

- The dataset class `TotalSegmentator` supports download the data on runtime with the initialized argument
`download: bool = True`. 
- For the multilabel classification task, every mask with at least one positive pixel is gets the label "1", all others get the label "0".
- For the multilabel classification task, the `TotalSegmentator` class creates a manifest file with one row/slice and the columns: `path`, `slice`, `split` and additional 117 columns for each class.
- The 3D images are treated as 2D. Every 25th slice is sampled and treated as individual image
- The splits with the following sizes are created after ordering images by filename:

| Splits | Train     | Validation | Test      |
|---|-----------|------------|-----------|
| #Samples | 737 (60%) | 246 (20%)  | 245 (20%) |


## Relevant links

* [TotalSegmentator dataset on zenodo](https://zenodo.org/records/10047292)
* [TotalSegmentator small subset (102 images) on zenodo](https://zenodo.org/records/10047263)

## License

[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode)
