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

## Release Process

The project uses semantic versioning. Releases are managed through GitHub Actions workflows.

### Making a Release

#### Option 1: Using GitHub Actions Workflow (Recommended)

1. Go to the GitHub repository
2. Navigate to "Actions" -> "Release" workflow
3. Click "Run workflow" 
4. Select the desired version increment type (patch, minor, or major)
5. Indicate whether this is a pre-release
6. Click "Run workflow"

The workflow will:
1. Calculate the new version number
2. Run quality checks and tests
3. Build and test the package
4. Update the version in pyproject.toml
5. Generate a changelog from commits since the last release
6. Create a GitHub release with assets
7. Deploy documentation for the new version
8. Publish the package to PyPI
9. Create a pull request to update the version in the main branch

#### Option 2: Local Release Preparation

You can prepare a release locally using the `release` command:

```bash
# For a patch release (0.2.0 -> 0.2.1)
pdm release patch

# For a minor release (0.2.1 -> 0.3.0)
pdm release minor

# For a major release (0.3.0 -> 1.0.0)
pdm release major

# For a specific version
pdm release to 1.2.3
```

This will:
1. Bump the version in pyproject.toml
2. Run lint checks and tests
3. Build the package
4. Create a git commit with the new version
5. Create a git tag for the release

After running this command, you'll need to:
1. Push the commit: `git push origin HEAD`
2. Push the tag to trigger the release: `git push origin v{version}`

#### Option 3: Manual Tag Creation

You can also create a release by manually creating and pushing a tag:

```bash
# Create an annotated tag
git tag -a v0.3.0 -m "Release v0.3.0"

# Push the tag
git push origin v0.3.0
```

When a tag is pushed, the "Tag Release" GitHub Actions workflow will automatically:
1. Extract the version from the tag
2. Update the version in pyproject.toml
3. Generate a changelog
4. Build and test the package
5. Create a GitHub release
6. Deploy documentation
7. Publish to PyPI
