# PatchCamelyon (PCam)

| Dataset | #Classes | #Patches | Patch Size | FoV (μm/px) | Task | Cancer Type |
|---|---|---|---|---| ---| ---|
| PCam | 2 | 327,680 | 96x96 | 10x (1.0) \* | Classification | Breast |

\* The slides were acquired and digitized at 2 different centres using a 40x objective but under-sampled to 10x to increase the field of view. Some papers do categorize it as 10x. Basically artificial 10x patches.

### Overview

The PatchCamelyon benchmark is a new and challenging image classification dataset. It consists of 327.680 color images (96 x 96px) extracted from histopathologic scans of lymph node sections. Each image is annotated with a binary label indicating presence of metastatic tissue.

### Splits

| Splits | Train | Validation | Test |
|---|---|---|---|
| PCam | 262,144 | 32,768 | 32,768 |


### Usage

#### Download
The dataset class supports download the data no runtime with the initialized argument
`download: bool = True`.

However, optionally, we do provide a script to download and extract the data and metadata.
To do so, from the library root path execute the following:
```sh
./assets/datasets/pcam/download.sh
```

### Citation
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
