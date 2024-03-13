# Installation

*Note: this section applies in the current form only to Kaiko-internal user testing and will be revised for the public package when publishing eva*


- Create and activate a virtual environment with Python 3.10+

- Install ***eva*** and the ***eva-vision*** package with:

```
pip install git+ssh://git@github.com/kaiko-ai/eva.git
pip install "eva[vision]"
```

- To be able to use the existing configs, you have to first download them from the [***eva*** GitHub repo](https://github.com/kaiko-ai/eva/tree/main):

    - Download the repo as zip file by clicking on `Code` > `Download ZIP`
    - Unzzip the file and copy the "config" folder into the directory where you installed eva


## Run ***eva***

Now you are all setup. You could run a complete ***eva*** workflow, for example with:
```
python -m eva fit --config configs/vision/dino_vit/online/bach.yaml 
```
This would:

 - Download and extract the dataset, if it has not been downloaded before.
 - Fit a model consisting of the frozen FM-backbone and a classification head on the train split.
 - Evaluate the trained model on the validation split and report the results.

However, before starting to run ***eva***, you might want to familiarize yourself with [How to use ***eva***](how_to_use.md) and then proceed to running ***eva*** with the [Tutorials](../tutorials/offline_vs_online.md)
