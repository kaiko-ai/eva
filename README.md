<div align="center">

<img src="./docs/images/eva-logo.png" width="400">

<br />

_Oncology FM Evaluation Framework by kaiko.ai_

[![PyPI](https://img.shields.io/pypi/v/pdm.svg?logo=python)](https://pypi.python.org/pypi/kaiko-eva)
[![CI](https://github.com/kaiko-ai/eva/workflows/CI/badge.svg)](https://github.com/kaiko-ai/eva/actions?query=workflow%3ACI)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg?labelColor=gray)](https://github.com/kaiko-ai/eva#license)


<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="https://kaiko-ai.github.io/eva/">Documentation</a> •
  <a href="https://kaiko-ai.github.io/eva/dev/datasets/">Datasets</a> •
  <a href="#contributing">Contribute</a>
</p>

</div>

<br />

`eva` is an evaluation framework for oncology foundation models (FMs) by [kaiko.ai](https://kaiko.ai/). Check out the [documentation](https://kaiko-ai.github.io/eva/) for more information.


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

# to install the expanded vision version
pip install 'kaiko-eva[vision]'

# to install everything
pip install 'kaiko-eva[all]'
```

To install the latest version of the `main` branch:
```sh
pip install "kaiko-eva[vision] @ git+https://github.com/kaiko-ai/eva.git"
```

You can verify that the installation was successful by executing:
```sh
eva --version
```

## How To Use

_eva_ can be used directly from the terminal as a CLI tool as follows:
```sh
eva {fit,predict,predict_fit} --config url/or/path/to/the/config.yaml 
```

For example, to perform a downstream evaluation of DINO ViT-S/16 on the BACH dataset with linear probing, by first infer the embeddings and perform 5 sequential fits, execute:
```sh
eva predict_fit --config https://raw.githubusercontent.com/kaiko-ai/eva/main/configs/vision/dino_vit/offline/bach.yaml
```

> [!NOTE] 
> All the datasets the support automatic download in the repo have by default the option to `false` and thus you manually have to set the argument to `true`, that is `download=true`.


To view all the possibles, execute:
```sh
eva --help
```

For more information, please refer to the [documentation](https://kaiko-ai.github.io/eva/dev/user-guide/tutorials/offline_vs_online/) and [tutorials](https://kaiko-ai.github.io/eva/dev/user-guide/advanced/replicate_evaluations/).


## Contributing

_eva_ is an open source project and welcomes contributions of all kinds. Please checkout the [developer](./docs/DEVELOPER_GUIDE.md) and [contributing guide](./docs/CONTRIBUTING.md) for help on how to do so.

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

---
<div align="center">
  <img src="./docs/images/kaiko-logo.png" width="200">
</div>
