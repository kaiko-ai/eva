# CoNSeP

CoNSep (Colorectal Nuclear Segmentation and Phenotypes) consists of 41 1000x1000 tiles extracted from 16 WSIs of unique patients. Labels are segmentation masks which indicate if a pixel belongs to one of 7 categories of cell nuclei. In total 24,319 unique nuclei are present.

## Raw data

### Key stats

|                       |                                                           |
|-----------------------|-----------------------------------------------------------|
| **Modality**          | Vision (WSI patches)                                      |
| **Task**              | Segmentation - 8 classes (non-"cell nucleus" + 7 categories)|
| **Data size**         | total: ~800MB                                             |
| **Image dimension**   | 1000 x 1000 x 3                                           |
| **Magnification (μm/px)**  | 40x (0.25)                                           |
| **Files format**      | `.png` images / `.mat` segmentation masks                 |
| **Number of images**  | 41                                                        |
| **Splits in use**     | Train and Test                                            |


### Organization

The data is organized as follows:

```
consep
├── Train
│   ├── Images                 # raw training input images
│   │   ├── train_1.png
│   │   └── ...
│   ├── Labels                 # train segmentation labels        
│   │   ├── train_1.mat
│   │   └── ...
│   ├── Overlay                # train images with bounding boxes, not in use
├── Test
│   ├── Images                 # raw test input images
│   │   ├── test_1.png
│   │   └── ...
│   ├── Labels                 # test segmentation labels        
│   │   ├── test_1.mat
│   │   └── ...
│   ├── Overlay                # test images with bounding boxes, not in use
└── README.txt                 # data description
```

## Download and preprocessing

*Note that the CoNSeP dataset is currently not available for download. As soon as it becomes availble we will add support & instructions (monitor [this issue](https://github.com/vqdang/hover_net/issues/267#issuecomment-2161334382) for updates)*

### Splits

We work with the splits provided by the data source. Since no "validation" split is provided, we use the "test" split as validation split.

| Splits   | Train           | Validation   | 
|----------|-----------------|--------------|
| #Samples | 27 (66%) | 14 (34%) | 


## Relevant links

* [CoNSeP Dataset description](https://paperswithcode.com/dataset/consep)
* [Data download](https://warwick.ac.uk/TIA/data/hovernet/) (currently not available)
* [GitHub issue for data availability](https://github.com/vqdang/hover_net/issues/267#issuecomment-2161334382)

## License

[TBD]
