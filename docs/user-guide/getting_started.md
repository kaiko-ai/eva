# Getting Started

## Installing eva
*Note: this section will be revised for the public package when publishing eva*


### Download the eva repo

First, install GIT LFS on your machine (see [reference](https://git-lfs.com/)) which is used to track assets, 
such as sample images used for tests.
```
brew install git lfs
```
Navigate to the directory where you'd like to install *eva* and install git-lfs:
```
git lfs install
```
Now clone the repo:
```
git clone git@github.com:kaiko-ai/eva.git
```

### Setup the environment and dependencies

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
Now you are ready to explore [How to use eva](how_to_use.md) 
