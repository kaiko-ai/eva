# Replicate results

On the [***eva*** main page](../../index.md) we present a table with evaluation results. All of these can be replicated with ***eva*** (see [Replicate Evaluations](../replicate_evaluations.md)).

In this tutorial, we replicate the results of the `DINO ViT-S16` model, randomly initialized and pretrained from ImageNet, with the [BACH](../../datasets/bach.md) classification task.

*Note: Feel free to use any of the other supported tasks - [CRC](../../datasets/crc.md), [MHIST](../../datasets/mhist.md) or [PatchCamelyon](../../datasets/patch_camelyon.md) - instead. However, be aware that running times might be significanlty longer since you will not benefit from already downloaded data and precomputed embeddins from the tutorial [***eva*** subcommands](eva_subcommands.md).*


## 1. Evaluate DINO ViT-S16 (random weights)

Set the environment variables:
```
export MAX_STEPS=12500
export BATCH_SIZE=256
export PRETRAINED=false
export EMBEDDINGS_DIR="./data/embeddings/dino_vits16_random/bach"
export DINO_BACKBONE=dino_vits16
export CHECKPOINT_PATH=null
export NORMALIZE_MEAN=[0.485,0.456,0.406]
export NORMALIZE_STD=[0.229,0.224,0.225]
```

If you already completed *Step 1* of the tutorial [***eva*** subcommands](eva_subcommands.md), you should have the embeddings for this configuration computed and stored in `./data/embeddings/dino_vits16/bach`. If so, we don't need to recompute the embeddings and can directly run the `fit`-command:
```
python -m eva fit --config configs/vision/dino_vit/offline/bach.yaml
```
However, if you did **not** compute the embeddings for the specified FM before, you will need to run the complete workflow with `predict_fit`-command instead:
```
python -m eva predict_fit --config configs/vision/dino_vit/offline/bach.yaml
```
Once the session completes, check the results in `logs/dino_vits16/offline/bach/<session-id>/results.json`. Can you confirm that the `MulticlassAccuracy` matches the one reported on the [***eva*** main page](../../index.md) for DINO ViT-S16 (N/A)?


## 2. Evaluate DINO ViT-S16 (ImageNet)

Lets also replicate the results of DINO ViT-S16 pretrained from ImageNet.

Again, first set the environment variables:
```
export MAX_STEPS=12500
export BATCH_SIZE=256
export PRETRAINED=true
export EMBEDDINGS_DIR="./data/embeddings/dino_vits16_imagenet/bach"
export DINO_BACKBONE=dino_vits16
export CHECKPOINT_PATH=null
export NORMALIZE_MEAN=[0.485,0.456,0.406]
export NORMALIZE_STD=[0.229,0.224,0.225]
```

Similar to *Step 1* above, you should already have computed and stored the embeddings if you ran *Step 3* of the tutorial [***eva*** subcommands](eva_subcommands.md). If this is the case run:
```
python -m eva fit --config configs/vision/dino_vit/offline/bach.yaml
```
Otherwise, run:
```
python -m eva predict --config configs/vision/dino_vit/offline/bach.yaml
```
Again, once the session completes, check the results in `logs/dino_vits16/offline/bach/<session-id>/results.json` and see if you can confirm that the `MulticlassAccuracy` matches the one reported on the [***eva*** main page](../../index.md) for DINO ViT-S16 (ImageNet)?