# Datasets

**eva** provides out-of-the-box support for several public datasets. When possible, the corresponding dataset classes facilitate automatic download to disk, if not possible, this documentation provides download instructions.

## Vision Datasets Overview

### Whole Slide (WSI) and microscopy image datasets

| Dataset |  #Patches | Patch Size | FoV (μm/px) | Task | Cancer Type | Download provided
|---|---|---|---|---| ---| ---| 
| [BACH](bach.md) | 400 | 2048x1536 | 20x (0.5) | Classification (4 classes) | Breast | Yes |
| [PatchCamelyon](patch_camelyon.md) | 327,680 | 96x96 | 10x (1.0) \* | Classification (2 classes) | Breast | Yes |

\* The slides were acquired and digitized at 2 different centres using a 40x objective but under-sampled to 10x to increase the field of view. Some papers do categorize it as 10x.


### Radiology datasets

| Dataset | #Images | Image Size | Task  | Download provided
|---|---|---|---|---|
| [TotalSegmentator](total_segmentator.md) | 1228 | ~300 x ~300 x ~350 \* |  Multilabel Classification (117 classes) | Yes |

\* 3D images of varying sizes