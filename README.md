<div align="center">

<img src="./docs/images/eva-logo.png" width="400">

<br />

_Oncology FM Evaluation Framework by kaiko.ai_


<a href="https://www.apache.org/licenses/LICENSE-2.0">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square" />
</a>

<br />
<br />

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#datasets">Datasets</a> •
  <a href="#contributing">Contribute</a>
</p>

</div>

---

### _About_

`eva` is [kaiko.ai](https://kaiko.ai/)'s evaluation framework for oncology foundation models (FMs). Check out the documentation (LINK TO BE ADDED) for more information.


## Installation

*Note: this section will be revised for the public package when publishing eva*

- Create and activate a virtual environment with Python 3.10+

- Install *eva* and the *eva-vision* package with:

```
pip install 'kaiko-eva[vision]'
```

- To be able to use the existing configs, download from the [*eva* GitHub repo](https://github.com/kaiko-ai/eva/tree/main) and move them to directory where you installed *eva*. 

### Run *eva*

Now you can run a complete *eva* workflow, for example with:
```
eva fit --config configs/vision/dino_vit/online/bach.yaml 
```
This will:

 - Download and extract the dataset, if it has not been downloaded before.
 - Fit a model consisting of the frozen FM-backbone and a classification head on the train split.
 - Evaluate the trained model on the validation split and report the results.

For more information, documentation and tutorials, refer to the documentation (LINK TO BE ADDED).

## Datasets

The following datasets are supported natively:

### Vision

#### Patch-level pathology datasets:
  - [BACH](./docs/datasets/bach.md)
  - [CRC](./docs/datasets/crc.md)
  - [MHIST](./docs/datasets/mhist.md)
  - [PatchCamelyon](./docs/datasets/patch_camelyon.md)

#### Radiology datasets:
  - [TotalSegmentator](./docs/datasets/total_segmentator.md)

## Contributing

_eva_ is an open source project and welcomes contributions of all kinds. Please checkout the [developer](./docs/DEVELOPER_GUIDE.md) and [contributing guide](./docs/CONTRIBUTING.md) for help on how to do so.

All contributors must follow the [code of conduct](./docs/CODE_OF_CONDUCT.md).

---
<div align="center">
  <img src="./docs/images/kaiko-logo.png" width="200">
</div>
