# Developer Guide

## Installation

To contribute any feature to EVA, you must install from source.

1. Install package and dependency manager PDM following the instructions [here](https://pdm-project.org/latest/#other-installation-methods).
2. Run `pdm install -G "dev"` to install the development dependencies. This will create a virtual environment in `eva/.venv`.
3. Run `pdm run python` to run a Python script or start an interactive Python shell within the context of the PDM-managed project environment. Alternatively, you can also activate the venv manually with `source .venv/bin/activate`.

PDM quick guide:
- `pdm lock -G:all` to generate or update the pdm.lock file including all groups.
- `pdm install -G:all`	to install all groups locked in the lockfile.
- `pdm update -G:all` to update all dependencies in the lock file.
- `pdm add <package-name>` to add a package.
- `pdm remove <package-name>` to remove a package.
For more information about managing dependencies please look [here](https://pdm-project.org/latest/usage/dependency/#manage-dependencies).

## Continuous Integration (CI)

For testing automation, we use [`nox`](https://nox.thea.codes/en/stable/index.html).

Installation:
- with brew: `brew install nox`
- with pip: `pip install --user --upgrade nox` (this way, you might need to run nox commands with `python -m nox` or specify an alias)

Commands:
- `nox` to run all the automation tests. 
- `nox -s fmt` to run the code formatting tests.
- `nox -s lint` to run the code lining tests.
- `nox -s check` to run the type-annotation tests.
- `nox -s test` to run the unit tests.
