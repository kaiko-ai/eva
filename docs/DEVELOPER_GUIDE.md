# Developer Guide


## Setting up a DEV environment

We use [PDM](https://pdm-project.org/latest/) as a package and dependency manager.
You can set up a local Python environment for development as follows: 
1. Install package and dependency manager PDM following the instructions [here](https://pdm-project.org/latest/#other-installation-methods).
2. Install system dependencies
    - For MacOS: `brew install Cmake`
    - For Linux (Debian): `sudo apt-get install build-essential cmake`
3. Run `PDM_PYTHON=$(pyenv which python) && pdm install -G all -G dev` to install the Python dependencies. This will create a virtual environment in `eva/.venv`. If you don't use `pyenv` to manage your python installations, you can replace `$(pyenv which python)` with the path to your python executable. Note that the python version used should match `PYTHON_VERSIONS` in `noxfile.py`, as this is the version is used for running the unit tests.

## Adding new dependencies 

Add a new dependency to the `core` submodule:<br>
`pdm add <package_name>`

Add a new dependency to the `vision` submodule:<br>
`pdm add -G vision -G all <package_name>`

For more information about managing dependencies please look [here](https://pdm-project.org/latest/usage/dependency/#manage-dependencies).

## Update dependencies
To update all dependencies in the lock file:
`pdm update`

To update the dependencies in a specific group
`pdm update -G <group_name>`

To update a specific dependency in a specified group
`pdm update -G <group_name> <package_name>`

## Regenerate the lock file
If you want to regenerate the lock file from scratch:
`pdm lock -G all -G vision -G lint -G typecheck -G test -G dev -G docs`

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
  - `nox -s test -- tests/eva/metrics/test_average_loss.py` to run specific tests
