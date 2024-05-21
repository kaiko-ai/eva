# Datasets

*eva* provides native support for several public datasets. When possible, the corresponding dataset classes facilitate automatic download to disk, if not possible, this documentation provides download instructions.

## Vision Datasets Overview

### Whole Slide (WSI) and microscopy image datasets

#### Slide-level
| Dataset                            | #Slides  | Slide Size                | Magnification (μm/px)  | Task                       | Cancer Type      |
|------------------------------------|----------|---------------------------|------------------------|----------------------------|------------------|
| [Camelyon16](camelyon16.md)        | 3,152    | ~100-250k x ~100-250k x 3 |  40x (0.25)            | Classification (2 classes) | Breast         |
| [PANDA](panda.md)                  | 3,152    | ~20k x 20k x 3            |  20x (0.5)             | Classification (6 classes) | Prostate         |


#### Patch-level
| Dataset                            | #Patches | Patch Size | Magnification (μm/px)  | Task                       | Cancer Type      |
|------------------------------------|----------|------------|------------------------|----------------------------|------------------|
| [BACH](bach.md)                    | 400      | 2048x1536  | 20x (0.5)              | Classification (4 classes) | Breast           |
| [CRC](crc.md)                      | 107,180  | 224x224    | 20x (0.5)              | Classification (9 classes) | Colorectal       |
| [PatchCamelyon](patch_camelyon.md) | 327,680  | 96x96      | 10x (1.0) \*           | Classification (2 classes) | Breast           |
| [MHIST](mhist.md)                  | 3,152    | 224x224    |  5x (2.0) \*           | Classification (2 classes) | Colorectal Polyp |

\* Downsampled from 40x (0.25 μm/px) to increase the field of view.

### Radiology datasets

| Dataset | #Images | Image Size | Task  | Download provided
|---|---|---|---|---|
| [TotalSegmentator](total_segmentator.md) | 1228 | ~300 x ~300 x ~350 \* |  Multilabel Classification (117 classes) | Yes |

\* 3D images of varying sizes
