<div align="center">

<img src="./docs/images/eva-logo.png" width="400">

<br />

_Oncology FM Evaluation Framework by KAIKO.ai_


<a href="https://www.python.org/">
  <img src="https://img.shields.io/badge/-python-blue?logo=python&logoColor=white&style=flat-square" />
</a>
<a href="https://lightning.ai/docs/pytorch/stable/">
  <img src="https://img.shields.io/badge/⚡️ lightning-792ee5?logo=pytorchlightning&logoColor=white&style=flat-square" />
</a>
<a href="https://github.com/wntrblm/nox">
  <img src="https://img.shields.io/badge/%F0%9F%A6%8A  nox-D85E00?style=flat-square" />
</a>
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

`eva` is [kaiko.ai](https://kaiko.ai/)'s evaluation framework for oncology foundation models (FMs).

## Installation

*Note: this section will be revised for the public package when publishing eva*


### Download the eva repo

First, make sure [GIT LFS](https://git-lfs.com/), which is used to track assets, 
such as sample images used for tests, is installed on your machine (instructions [here](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)).

Now, clone the repo:
```
git clone git@github.com:kaiko-ai/eva.git
```

### Environment and dependencies

Now install *eva* and it's dependencies in a virtual environment. This can be done with the Python 
package and dependency manager PDM (see [documentation](https://pdm-project.org/latest/)).

Install PDM on your machine:
```
brew install pdm
```
Navigate to the eva root directory and run:
```
pdm install
```
This will install eva and all its dependencies in a virtual environment. Activate the venv with:
```
source .venv/bin/activate
```
Now you are ready to start! Start the documentation
```
mkdocs serve
```
and explore it in your [browser](http://127.0.0.1:8000/). Read through the main page and navigate
to [how-to-use]http://127.0.0.1:8000/user-guide/how_to_use/ to run *eva*

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

EVA is an open source project and welcomes contributions of all kinds. Please checkout the [developer](./docs/DEVELOPER_GUIDE.md) and [contributing guide](./docs/CONTRIBUTING.md) for help on how to do so.

All contributors must follow the [code of conduct](./docs/CODE_OF_CONDUCT.md).

---
<div align="center">
  <img src="./docs/images/kaiko-logo.png" width="200">
</div>
