# MHIST

MHIST is a binary classification task which comprises of 3,152 hematoxylin and eosin (H&E)-stained
Formalin Fixed Paraffin-Embedded (FFPE) fixed-size images (224 by 224 pixels) of colorectal polyps
from the Department of Pathology and Laboratory Medicine at Dartmouth-Hitchcock Medical Center (DHMC).

The tissue classes are: Hyperplastic Polyp (HP), Sessile Serrated Adenoma (SSA). This classification task
focuses on the clinically-important binary distinction between HPs and SSAs, a challenging problem with
considerable inter-pathologist variability. HPs are typically benign, while sessile serrated adenomas are
precancerous lesions that can turn into cancer if left untreated and require sooner follow-up examinations.
Histologically, HPs have a superficial serrated architecture and elongated crypts, whereas SSAs are characterized
by broad-based crypts, often with complex structure and heavy serration.


## Raw data

### Key stats

|                                |                                                     |
|--------------------------------|-----------------------------------------------------|
| **Modality**                   | Vision (WSI patches)                                |
| **Task**                       | Binary classification (2 classes)                   |
| **Cancer type**                | Colorectal Polyp                                    |
| **Data size**                  | 354 MB                                              |
| **Image dimension**            | 224 x 224 x 3                                       |
| **Magnification (μm/px)**      | 5x (2.0) \*                                         |
| **Files format**               | `.png` images                                       |
| **Number of images**           | 3,152 (2,175 train, 977 test)                       |
| **Splits in use**              | annotations.csv (train / test)                      |

\* Downsampled from 40x to increase the field of view.

### Organization

The contents from `images.zip` and the file `annotations.csv` from [bmirds](https://bmirds.github.io/MHIST/#accessing-dataset) are organized as follows:

```
mhist                           # Root folder
├── images                      # All the dataset images
│   ├── MHIST_aaa.png
│   ├── MHIST_aab.png
│   ├── ...
└── annotations.csv             # The dataset annotations file
```

## Download and preprocessing

To download the dataset, please visit the [access portal on BMIRDS](https://bmirds.github.io/MHIST/#accessing-dataset)
and follow the instructions. You will then receive an email with all the relative links that you can use to download
the data (`images.zip`, `annotations.csv`, `Dataset Research Use Agreement.pdf` and `MD5SUMs.txt`). 

Please create a root folder, e.g. `mhist`, and download all the files there, which unzipping the contents of
`images.zip` to a directory named `images` inside your root folder (i.e. `mhist/images`). Afterwards, you can
(optionally) delete the `images.zip` file.

### Splits

We work with the splits provided by the data source. Since no "validation" split is provided, we use the "test" split as validation split.

 - Train split: `annotations.csv` :: "Partition" == "train"
 - Validation split: `annotations.csv` :: "Partition" == "test"

| Splits   | Train           | Validation   | 
|----------|-----------------|--------------|
| #Samples | 2,175 (69%)     | 977 (31%)    | 

## Relevant links

* [Accessing MHIST Dataset (BMIRDS)](https://bmirds.github.io/MHIST/#accessing-dataset)
* [Paper: A Petri Dish for Histopathology Image Analysis](https://arxiv.org/pdf/2101.12355.pdf)
