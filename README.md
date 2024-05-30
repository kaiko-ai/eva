<div align="center">

<br />

<img src="https://github.com/kaiko-ai/eva/blob/main/docs/images/eva-logo.png?raw=true" width="340">

<br />
<br />

_Oncology FM Evaluation Framework by kaiko.ai_

[![PyPI](https://img.shields.io/pypi/v/kaiko-eva.svg?logo=python)](https://pypi.python.org/pypi/kaiko-eva)
[![docs](https://img.shields.io/badge/📚_docs-latest-green)](https://kaiko-ai.github.io/eva/latest)
[![license](https://img.shields.io/badge/⚖️_License-Apache%202.0-blue.svg?labelColor=gray)](https://github.com/kaiko-ai/eva#license)<br>
[![paper](http://img.shields.io/badge/OpenReview-MIDL_2024-B31B1B.svg)](https://openreview.net/forum?id=FNBQOPj18N&noteId=FNBQOPj18N)

<p align="center">
  <a href="https://github.com/kaiko-ai/eva#installation">Installation</a> •
  <a href="https://github.com/kaiko-ai/eva#how-to-use">How To Use</a> •
  <a href="https://kaiko-ai.github.io/eva/">Documentation</a> •
  <a href="https://kaiko-ai.github.io/eva/dev/datasets/">Datasets</a> •
  <a href="https://github.com/kaiko-ai/eva#benchmarks">Benchmarks</a> <br>
  <a href="https://github.com/kaiko-ai/eva#contributing">Contribute</a> •
  <a href="https://github.com/kaiko-ai/eva#acknowledgements">Acknowledgements</a>
</p>

</div>

<br />

_`eva`_ is an evaluation framework for oncology foundation models (FMs) by [kaiko.ai](https://kaiko.ai/).
Check out the [documentation](https://kaiko-ai.github.io/eva/) for more information.

### Highlights:
- Easy and reliable benchmark of Oncology FMs
- Automatic embedding inference and evaluation of a downstream task
- Native support of popular medical [datasets](https://kaiko-ai.github.io/eva/dev/datasets/) and models
- Produce statistics over multiple evaluation fits and multiple metrics

## Installation

Simple installation from PyPI:
```sh
# to install the core version only
pip install kaiko-eva

# to install the expanded `vision` version
pip install 'kaiko-eva[vision]'

# to install everything
pip install 'kaiko-eva[all]'
```

To install the latest version of the `main` branch:
```sh
pip install "kaiko-eva[all] @ git+https://github.com/kaiko-ai/eva.git"
```

You can verify that the installation was successful by executing:
```sh
eva --version
```

## How To Use

_`eva`_ can be used directly from the terminal as a CLI tool as follows:
```sh
eva {fit,predict,predict_fit} --config url/or/path/to/the/config.yaml 
```

When used as a CLI tool, _`eva`_ supports configuration files (`.yaml`) as an argument to define its functionality.
Native supported configs can be found at the [configs](https://github.com/kaiko-ai/eva/tree/main/configs) directory
of the repo. Apart from cloning the repo, you can download the latest config folder as `.zip` from your browser from
[here](https://download-directory.github.io/?url=https://github.com/kaiko-ai/eva/tree/main/configs). Alternatively,
from a specific release the configs can be downloaded from the terminal as follows:
```sh
curl -LO https://github.com/kaiko-ai/eva/releases/download/0.0.1/configs.zip | unzip configs
```

For example, to perform a downstream evaluation of DINO ViT-S/16 on the BACH dataset with
linear probing by first inferring the embeddings and performing 5 sequential fits, execute:
```sh
# from a locally stored config file
eva predict_fit --config ./configs/vision/dino_vit/offline/bach.yaml

# from a remote stored config file
eva predict_fit --config https://raw.githubusercontent.com/kaiko-ai/eva/main/configs/vision/dino_vit/offline/bach.yaml
```

> [!NOTE] 
> All the datasets that support automatic download in the repo have by default the option to automatically download set to false.
> For automatic download you have to manually set download=true.


To view all the possibles, execute:
```sh
eva --help
```

For more information, please refer to the [documentation](https://kaiko-ai.github.io/eva/dev/user-guide/tutorials/offline_vs_online/)
and [tutorials](https://kaiko-ai.github.io/eva/dev/user-guide/advanced/replicate_evaluations/).

## Benchmarks

In this section you will find model benchmarks which were generated with _`eva`_.

### Table I: WSI patch-level benchmark

<br />

<div align="center">

| Model                                            | BACH  | CRC   | MHIST | PCam/val | PCam/test |
|--------------------------------------------------|-------|-------|-------|----------|-----------|
| ViT-S/16 _(random)_	<sup>[1]</sup>               | 0.410 | 0.617 | 0.501 | 0.753    | 0.728     |
| ViT-S/16 _(ImageNet)_ <sup>[1]</sup>             | 0.695 | 0.935 | 0.831 | 0.864    | 0.849     |
| ViT-B/8 _(ImageNet)_ <sup>[1]</sup>              | 0.710 | 0.939 | 0.814 | 0.870    | 0.856     |
| ViT-L/14 _(ImageNet)_ <sup>[1]</sup>             | 0.707 | 0.916 | 0.832 | 0.873    | 0.888     |
| DINO<sub>(p=16)</sub> <sup>[2]</sup>             | 0.801 | 0.934 | 0.768 | 0.889    | 0.895     |
| Phikon <sup>[3]</sup>                            | 0.725 | 0.935 | 0.777 | 0.912    | 0.915     |
| UNI <sup>[4]</sup>                               | 0.814 | 0.950 | 0.837 | 0.936    | 0.938     |
| ViT-S/16 _(kaiko.ai)_ <sup>[5]</sup>             | 0.797 | 0.943 | 0.828 | 0.903    | 0.893     |
| ViT-S/8 _(kaiko.ai)_ <sup>[5]</sup>              | 0.834 | 0.946 | 0.832 | 0.897    | 0.887     |
| ViT-B/16 _(kaiko.ai)_	<sup>[5]</sup>             | 0.810 | 0.960 | 0.826 | 0.900    | 0.898     |
| ViT-B/8 _(kaiko.ai)_ <sup>[5]</sup>              | 0.865 | 0.956 | 0.809 | 0.913    | 0.921     |
| ViT-L/14 _(kaiko.ai)_ <sup>[5]</sup>             | 0.870 | 0.930 | 0.809 | 0.908    | 0.898     |

_Table I: Linear probing evaluation of FMs on patch-level downstream datasets.<br> We report averaged balanced accuracy
over 5 runs, with an average standard deviation of ±0.003._

</div>

<br />

_References_:
1. _"Emerging properties in self-supervised vision transformers”_, [arXiv](https://arxiv.org/abs/2104.14294)
2. _"Benchmarking self-supervised learning on diverse pathology datasets”_, [arXiv](https://arxiv.org/abs/2212.04690)
3. _"Scaling self-supervised learning for histopathology with masked image modeling”_, [medRxiv](https://www.medrxiv.org/content/10.1101/2023.07.21.23292757v1)
4. _"A General-Purpose Self-Supervised Model for Computational Pathology”_, [arXiv](https://arxiv.org/abs/2308.15474)
5. _"Towards Training Large-Scale Pathology Foundation Models: from TCGA to Hospital Scale”_, [arXiv](https://arxiv.org/pdf/2404.15217)

## Contributing

_`eva`_ is an open source project and welcomes contributions of all kinds. Please checkout the [developer](./docs/DEVELOPER_GUIDE.md)
and [contributing guide](./docs/CONTRIBUTING.md) for help on how to do so.

All contributors must follow the [code of conduct](./docs/CODE_OF_CONDUCT.md).


## Acknowledgements

Our codebase is built using multiple opensource contributions

<div align="center">

[![python](https://img.shields.io/badge/-Python-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-⚡️_Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)<br>
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)<br>
[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm-project.org)
[![Nox](https://img.shields.io/badge/%F0%9F%A6%8A-Nox-D85E00.svg)](https://github.com/wntrblm/nox)
[![Built with Material for MkDocs](https://img.shields.io/badge/Material_for_MkDocs-526CFE?logo=MaterialForMkDocs&logoColor=white)](https://squidfunk.github.io/mkdocs-material/)

</div>


## Citation

If you find this repository useful, please consider giving a star ⭐ and adding the following citation:

```
@inproceedings{
kaiko.ai2024eva,
title={eva: Evaluation framework for pathology foundation models},
author={kaiko.ai and Ioannis Gatopoulos and Nicolas K{\"a}nzig and Roman Moser and Sebastian Ot{\'a}lora},
booktitle={Medical Imaging with Deep Learning},
year={2024},
url={https://openreview.net/forum?id=FNBQOPj18N}
}
```

<br />

<div align="center">
  <img src="https://github.com/kaiko-ai/eva/blob/main/docs/images/kaiko-logo.png?raw=true" width="200">
</div>
