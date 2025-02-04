# UniToPatho


UniToPatho is an annotated dataset of 9536 hematoxylin and eosin stained patches extracted from 292 whole-slide images, meant for training deep neural networks for colorectal polyps classification and adenomas grading. The slides are acquired through a Hamamatsu Nanozoomer S210 scanner at 20x magnification (0.4415 μm/px). Each slide belongs to a different patient and is annotated by expert pathologists, according to six classes as follows:

- NORM - Normal tissue;
- HP - Hyperplastic Polyp;
- TA.HG - Tubular Adenoma, High-Grade dysplasia;
- TA.LG - Tubular Adenoma, Low-Grade dysplasia;
- TVA.HG - Tubulo-Villous Adenoma, High-Grade dysplasia;
- TVA.LG - Tubulo-Villous Adenoma, Low-Grade dysplasia.

For this benchmark we used only the `800` subset which contains 8669 images of resolution 1812x1812 (the `7000` subset contains much bigger images and would therefore be difficult to handle as patch classification task).

## Raw data

### Key stats

|                                |                             |
|--------------------------------|-----------------------------|
| **Modality**                   | Vision (WSI patches)        |
| **Task**                       | Multiclass classification (6 classes) |
| **Cancer type**                | Colorectal                  |
| **Data size**                  | 48.37 GB                    |
| **Image dimension**            | 1812 x 1812                 |
| **Magnification (μm/px)**      | 20x (0.4415)                |
| **Magnification after resize (μm/px)**      | 162x (3.57)    |
| **Files format**               | `png`                       |
| **Number of images**           | 8669                        |


### Splits

The data source provides train/validation splits

| Splits | Train          | Validation    |
|----------|--------------|---------------|
| #Samples | 6270 (72.33) | 2399 (27.67%) |

The dataset authors only provide two splits, which is why we don't report performance on a third test split.


### Organization

The UniToPatho data is organized as follows (note that we are using only the `800` subset):

```
unitopatho
├── 800
    test.csv
    train.csv
│   ├── HP                    # 1 folder per class 
│   ├── NORM
│   ├── TA.HG
│   ├── ...
```


## Download and preprocessing
The `UniToPatho` dataset class doesn't download the data during runtime and must be downloaded manually from [the official source](https://ieee-dataport.org/open-access/unitopatho).

## Relevant links

* [GitHub Repo](https://github.com/EIDOSLAB/UNITOPATHO)

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)