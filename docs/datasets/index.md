# Datasets

*eva* provides native support for several public datasets. When possible, the corresponding dataset classes facilitate automatic download to disk, if not possible, this documentation provides download instructions.

## Vision Datasets Overview

### Whole Slide (WSI) and microscopy image datasets

#### Patch-level
| Dataset                            | #Patches | Patch Size  | Magnification (μm/px)  | Task                       |   Tissue Type   |
|------------------------------------|----------|-------------|------------------------|----------------------------|------------------|
| [BACH](bach.md)                    | 400      | 2048x1536   | 20x (0.5)              | Classification (4 classes) | Breast           |
| [BRACS](bracs.md)                  | 4539     | variable   | 40x (0.25)             | Classification (7 classes) | Breast           |
| [BreakHis](breakhis.md)            | 1471     | 700x460    | 40x (0.25)             | Classification (4 classes) | Breast           |
| [CRC](crc.md)                      | 107,180  | 224x224     | 20x (0.5)              | Classification (9 classes) | Colorectal       |
| [GleasonArvaniti](crc.md)          | 22,752   | 750x750    | 40x (0.23)             | Classification (4 classes) | Prostate         |
| [PatchCamelyon](patch_camelyon.md) | 327,680  | 96x96       | 10x (1.0) \*           | Classification (2 classes) | Breast           |
| [MHIST](mhist.md)                  | 3,152    | 224x224     |  5x (2.0) \*           | Classification (2 classes) | Colorectal Polyp |
| [UniToPatho](unitopatho.md)          | 8669     | 1812 x 1812 |  20x (0.4415)          | Classification (6 classes) | Colorectal Polyp |
| [MoNuSAC](monusac.md)              | 294      | 113x81 - 1398x1956  | 40x (0.25)    | Segmentation (4 classes)   | Multi-Organ Cell Type (Breast, Kidney, Lung and Prostate) |
| [CoNSeP](consep.md)                | 41       | 1000x1000   |  40x (0.25) \*         | Segmentation (8 classes)   | Colorectal Nuclear |

\* Downsampled from 40x (0.25 μm/px) to increase the field of view.

#### Slide-level
| Dataset                            | #Slides  | Slide Size                | Magnification (μm/px)  | Task                       | Cancer Type      |
|------------------------------------|----------|---------------------------|------------------------|----------------------------|------------------|
| [Camelyon16](camelyon16.md)        | 400      | ~100-250k x ~100-250k x 3 |  40x (0.25)            | Classification (2 classes) | Breast           |
| [PANDA](panda.md)                  | 9,555    | ~20k x 20k x 3            |  20x (0.5)             | Classification (6 classes) | Prostate         |
| [PANDASmall](panda_small.md)       | 1,909     | ~20k x 20k x 3           |  20x (0.5)             | Classification (6 classes) | Prostate         |


### Radiology datasets

| Dataset | #Images | Image Size | Task  | Download provided
|---|---|---|---|---|
| [BTCV](btcv.md) | 30 | ~100 x 512 x 512 \* |  Semantic Segmentation (14 classes) | Yes |
| [LiTS17](lits17.md) | 130 | ~500 x 512 x 512 \* |  Semantic Segmentation (3 classes) | No |
| [MSDTask7Pancreas](msd_task7_pancreas.md) | 281 | ~100 x ~512 x ~512 \* |  Semantic Segmentation (3 classes) | Yes |

\* 3D images of varying sizes

## Language Datasets Overview

| Dataset | #Samples | Task | Domain | Download provided |
|---------|----------|------|---------|-------------------|
| [PubMedQA](pubmedqa.md) | 1,000 | Classification (3 classes) | Biomedical Q&A | Yes |
