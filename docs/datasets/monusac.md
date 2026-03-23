# MoNuSAC

MoNuSAC (Multi-Organ Nuclei Segmentation And Classification Challenge) consists of H&E stained tissue images of four organs with annotations of multiple cell-types including epithelial cells, lymphocytes, macrophages, and neutrophils with over 46,000 nuclei from 37 hospitals and 71 patients.

## Raw data

### Key stats

|                       |                                                           |
|-----------------------|-----------------------------------------------------------|
| **Modality**          | Vision (WSI patches)                                      |
| **Task**              | Segmentation - 5 classes *                               |
| **Data size**         | total: ~600MB                                             |
| **Image dimension**   | 113x81 - 1398x1956                                        |
| **Magnification (μm/px)**  | 40x (0.25)                                           |
| **Files format**      | `.svs` or `.tif` images / `.xml` segmentation masks       |
| **Number of images**  | 294                                                       |
| **Splits in use**     | Train and Test                                            |

\* The fith class is "ambiguous" and doesn't contain a specific cell type.

### Organization

The data is organized as follows:

```
monusac
├── MoNuSAC_images_and_annotations
│   ├── TCGA-5P-A9K0-01Z-00-DX1             # patient id
│   │   ├── TCGA-5P-A9K0-01Z-00-DX1_1.svs   # tissue image
│   │   ├── TCGA-5P-A9K0-01Z-00-DX1_1.tif   # tissue image
│   │   ├── TCGA-5P-A9K0-01Z-00-DX1_1.xml   # annotations
│   │   └── ...
├── MoNuSAC Testing Data and Annotations
│   ├── TCGA-5P-A9K0-01Z-00-DX1             # patient id
│   │   ├── TCGA-5P-A9K0-01Z-00-DX1_1.svs   # tissue image
│   │   ├── TCGA-5P-A9K0-01Z-00-DX1_1.tif   # tissue image
│   │   ├── TCGA-5P-A9K0-01Z-00-DX1_1.xml   # annotations
│   │   └── ...
```

## Download and preprocessing
The dataset class `MoNuSAC` supports downloading the data during runtime by setting the init argument `download=True`.

> [!NOTE]
> In the provided `MoNuSAC`-config files the download argument is set to `false`. To enable automatic download you will need to open the config and set `download: true`.


### Splits

We work with the splits provided by the data source. Since no "validation" split is provided, we use the "test" split as validation split.

| Splits   | Train           | Validation   | 
|----------|-----------------|--------------|
| #Samples | 209 (71%)       | 85 (29%)     | 


## Relevant links

* [MoNuSAC Dataset](https://monusac-2020.grand-challenge.org/Home/)

## License

The challenge data is released under the creative commons license ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)).
