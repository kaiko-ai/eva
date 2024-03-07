# Getting Started

*Note: this section applies in the current form only to Kaiko-internal user testing and will be revised for the public package when publishing eva*

## Installation


- Create and activate a virtual environment with Python 3.10+

- Install ***eva*** and the ***eva-vision*** package with:

```
pip install git+ssh://git@github.com/kaiko-ai/eva.git
pip install "eva[vision]"
```

- To be able to use the existing configs, you have to download them first

    - Go to [the ***eva*** GitHub repo](https://github.com/kaiko-ai/eva/tree/main)
    - Download the repo as zip file by clicking on `Code` > `Download ZIP`
    - Unzzip the file and copy the "config" folder into the directory where you installed eva


## Run ***eva***

Run a complete ***eva*** workflow with the:
```
python -m eva fit --config configs/vision/tests/online/bach.yaml 
```
This will:

 - Download and extract the BACH dataset to `./data}/bach`, if it has not been downloaded before.
 - Fit a complete model consisting of the frozen FM-backbone (a pretrained `dino_vits16`) and a downstream head (single layer MLP) on the BACH-train split.
 - Evaluate the trained model on the val split and report the results

To learn more about how to run ***eva*** and customize your runs, familiarize yourself with [How to use ***eva***](how_to_use.md) and get started with [Tutorials](tutorials.md) 
