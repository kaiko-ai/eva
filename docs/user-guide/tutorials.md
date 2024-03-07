# Tutorials


Before starting, download the configs for this tutorial from the [***eva*** GitHub repo](https://github.com/kaiko-ai/eva/tree/main):

1. Download the repo as zip file by clicking on `Code` > `Download ZIP`
2. Unzzip the file and copy the "config" folder into the directory where you installed eva


## 1. Run an *online*-evaluation

*Note: This step executes the same command & configuration as in the section "Getting started"*

Run a complete online workflow with the following command:
```
python -m eva fit --config configs/vision/dino_vit/online/bach.yaml
```

The `fit` run will:

 - Download and extract the BACH dataset to `./data/bach`, if it has not been downloaded before.
 - Fit a complete model consisting of the frozen FM-backbone (a pretrained `dino_vits16`) and a downstream head (single layer MLP) on the BACH-train split.
 - Evaluate the trained model on the val split and report the results

The running time will depend on your internet connection and compute setup. A complete run with internet speed ~200MB/s on a MacBook Pro with M1+ chip should take around 20 [**TBD**] minutes.

Once the run is complete:

 - check out some of the raw images in `<...>/eva/data/bach/ICIAR2018_BACH_Challenge` (this can already be done once the data download step is complete)
 - check out the evaluation results json file in `<...>/eva/logs/dino_vit/online/bach` [**TBD**]


## 2. Run a complete *offline*-evaluation

Now, run a complete offline workflow with the following command:
```
python -m eva predict_fit --config configs/vision/dino_vit/offline/patch_camelyon.yaml
```

The `predict_fit` run will:

 - Download and extract the BACH dataset to `./data}/bach`, if it has not been downloaded before. If you ran the *online*-evaluation above before, this step will be skipped.
 - ("predict") Computes the embeddings for all input images with the FM-backbone (a pretrained `dino_vits16`) and stores them in `./data/embeddings/bach` along with a `manifest.csv` file that keeps track of the mapping between input images and embeddings.
 - ("fit") Fit a downstream head (single layer MLP) on the BACH-train split, using the computed embeddings and provided labels as input.
 - Evaluate the trained model on the val split and report the results

Once the run is complete:

 - check out the evaluation results json file in `<...>/eva/logs/dino_vit/online/bach` [**TBD**]

 Note: comparing the results with the run from 2. you will notice a difference in performance. This is because we ran the online workflow with fewer epochs. Optionally, to verify that both workflows produce identical results, change the `max_steps` parameter in `configs/vision/dino_vit/offline/patch_camelyon.yaml` to [**TBD**], and rerun th `predict_fit` command above.

## 3. Run the fit step of the *offline*-evaluation

If you ran the complete *offline*-evaluation above, you have already computed and stored all the embeddings for the BACH dataset with the pretrained `dino_vits16` FM-backbone. (In case you skipped the previous step, generate them now by running `python -m eva predict --config configs/vision/dino_vit/offline/patch_camelyon.yaml`)

Now, run the fit step offline workflow with the following command with:
```
python -m eva fit --config configs/vision/dino_vit/offline/patch_camelyon.yaml
```

The *offline*-`fit` run will:

 - ("fit") Fit a downstream head (single layer MLP) on the BACH-train split, using the computed embeddings and provided labels as input.
 - Evaluate the trained model on the val split and report the results

Once the run is complete:

 - check out the evaluation results json file in `<...>/eva/logs/dino_vit/online/bach` [**TBD**], verify that the results are identical with those from the previous `predict_fit` run.
