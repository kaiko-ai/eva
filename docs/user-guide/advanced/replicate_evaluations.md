# Replicate evaluations

To produce the evaluation results presented [here](../../index.md#evaluation-results), you can run *eva* with the settings below.

Make sure to replace `<task>` in the commands below with `bach`, `crc`, `mhist` or `patch_camelyon`.

*Note that to run the commands below you will need to first download the data. [BACH](../../datasets/bach.md), [CRC](../../datasets/crc.md) and [PatchCamelyon](../../datasets/patch_camelyon.md) provide automatic download by setting the argument `download: true` (either modify the config-files or set the environment variable `DOWNLOAD=true`). In the case of MHIST you will need to download the data manually by following the instructions provided [here](../../datasets/mhist.md#download-and-preprocessing).*

## DINO ViT-S16 (random weights)

Evaluating the backbone with randomly initialized weights serves as a baseline to compare the pretrained FMs to an FM that produces embeddings without any prior learning on image tasks. To evaluate, run:

```
PRETRAINED=false \
EMBEDDINGS_ROOT="./data/embeddings/dino_vits16_random" \
eva predict_fit --config configs/vision/dino_vit/offline/<task>.yaml
```

## DINO ViT-S16 (ImageNet)

The next baseline model, uses a pretrained ViT-S16 backbone with ImageNet weights. To evaluate, run:

```
EMBEDDINGS_ROOT="./data/embeddings/dino_vits16_imagenet" \
eva predict_fit --config configs/vision/dino_vit/offline/<task>.yaml
```

## DINO ViT-B8 (ImageNet)

To evaluate performance on the larger ViT-B8 backbone pretrained on ImageNet, run:
```
EMBEDDINGS_ROOT="./data/embeddings/dino_vitb8_imagenet" \
DINO_BACKBONE=dino_vitb8 \
IN_FEATURES=768 \
eva predict_fit --config configs/vision/dino_vit/offline/<task>.yaml
```

## DINOv2 ViT-L14 (ImageNet)

To evaluate performance on Dino v2 ViT-L14 backbone pretrained on ImageNet, run:
```
PRETRAINED=true \
EMBEDDINGS_ROOT="./data/embeddings/dinov2_vitl14_kaiko" \
REPO_OR_DIR=facebookresearch/dinov2:main \
DINO_BACKBONE=dinov2_vitl14_reg \
FORCE_RELOAD=true \
IN_FEATURES=1024 \
eva predict_fit --config configs/vision/dino_vit/offline/<task>.yaml
```

## Lunit - DINO ViT-S16 (TCGA)

[Lunit](https://www.lunit.io/en), released the weights for a DINO ViT-S16 backbone, pretrained on TCGA data
on [GitHub](https://github.com/lunit-io/benchmark-ssl-pathology/releases/). To evaluate, run:

```
PRETRAINED=false \
EMBEDDINGS_ROOT="./data/embeddings/dino_vits16_lunit" \
CHECKPOINT_PATH="https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights/dino_vit_small_patch16_ep200.torch" \
NORMALIZE_MEAN=[0.70322989,0.53606487,0.66096631] \
NORMALIZE_STD=[0.21716536,0.26081574,0.20723464] \
eva predict_fit --config configs/vision/dino_vit/offline/<task>.yaml
```

## Owkin - iBOT ViT-B16 (TCGA)

[Owkin](https://www.owkin.com/) released the weights for "Phikon", an FM trained with iBOT on TCGA data, via
[HuggingFace](https://huggingface.co/owkin/phikon). To evaluate, run:

```
EMBEDDINGS_ROOT="./data/embeddings/dino_vitb16_owkin" \
eva predict_fit --config configs/vision/owkin/phikon/offline/<task>.yaml
```

Note: since *eva* provides the config files to evaluate tasks with the Phikon FM in 
"configs/vision/owkin/phikon/offline", it is not necessary to set the environment variables needed for
the runs above.

## UNI - DINOv2 ViT-L16 (Mass-100k)

The UNI FM, introduced in [[1]](#references) is available on [HuggingFace](https://huggingface.co/MahmoodLab/UNI). Note that access needs to 
be requested.

Unlike the other FMs evaluated for our leaderboard, the UNI model uses the vision library `timm` to load the model. To 
accomodate this, you will need to modify the config files (see also [Model Wrappers](model_wrappers.md)).

Make a copy of the task-config you'd like to run, and replace the `backbone` section with:
```
backbone:
    class_path: eva.models.ModelFromFunction
    init_args:
        path: timm.create_model
        arguments:
            model_name: vit_large_patch16_224
            patch_size: 16
            init_values: 1e-5
            num_classes: 0
            dynamic_img_size: true
        checkpoint_path: <path/to/pytorch_model.bin>
```

Now evaluate the model by running:
```
EMBEDDINGS_ROOT="./data/embeddings/dinov2_vitl16_uni" \
IN_FEATURES=1024 \
eva predict_fit --config path/to/<task>.yaml
```


## kaiko.ai - DINO ViT-S16 (TCGA)

To evaluate [kaiko.ai's](https://www.kaiko.ai/) FM with DINO ViT-S16 backbone, pretrained on TCGA data 
on [GitHub](https://github.com/lunit-io/benchmark-ssl-pathology/releases/), run:

```
PRETRAINED=false \
EMBEDDINGS_ROOT="./data/embeddings/dino_vits16_kaiko" \
CHECKPOINT_PATH=[TBD*] \
NORMALIZE_MEAN=[0.5,0.5,0.5] \
NORMALIZE_STD=[0.5,0.5,0.5] \
eva predict_fit --config configs/vision/dino_vit/offline/<task>.yaml
```

\* path to public checkpoint will be added when available.

## kaiko.ai - DINO ViT-S8 (TCGA)

To evaluate [kaiko.ai's](https://www.kaiko.ai/) FM with DINO ViT-S8 backbone, pretrained on TCGA data 
on [GitHub](https://github.com/lunit-io/benchmark-ssl-pathology/releases/), run:

```
PRETRAINED=false \
EMBEDDINGS_ROOT="./data/embeddings/dino_vits8_kaiko" \
DINO_BACKBONE=dino_vits8 \
CHECKPOINT_PATH=[TBD*] \
NORMALIZE_MEAN=[0.5,0.5,0.5] \
NORMALIZE_STD=[0.5,0.5,0.5] \
eva predict_fit --config configs/vision/dino_vit/offline/<task>.yaml
```

\* path to public checkpoint will be added when available.

## kaiko.ai - DINO ViT-B16 (TCGA)

To evaluate [kaiko.ai's](https://www.kaiko.ai/) FM with the larger DINO ViT-B16 backbone, pretrained on TCGA data,
run:

```
PRETRAINED=false \
EMBEDDINGS_ROOT="./data/embeddings/dino_vitb16_kaiko" \
DINO_BACKBONE=dino_vitb16 \
CHECKPOINT_PATH=[TBD*] \
IN_FEATURES=768 \
NORMALIZE_MEAN=[0.5,0.5,0.5] \
NORMALIZE_STD=[0.5,0.5,0.5] \
eva predict_fit --config configs/vision/dino_vit/offline/<task>.yaml
```

\* path to public checkpoint will be added when available.

## kaiko.ai - DINO ViT-B8 (TCGA)

To evaluate [kaiko.ai's](https://www.kaiko.ai/) FM with the larger DINO ViT-B8 backbone, pretrained on TCGA data,
run:

```
PRETRAINED=false \
EMBEDDINGS_ROOT="./data/embeddings/dino_vitb8_kaiko" \
DINO_BACKBONE=dino_vitb8 \
CHECKPOINT_PATH=[TBD*] \
IN_FEATURES=768 \
NORMALIZE_MEAN=[0.5,0.5,0.5] \
NORMALIZE_STD=[0.5,0.5,0.5] \
eva predict_fit --config configs/vision/dino_vit/offline/<task>.yaml
```

\* path to public checkpoint will be added when available.

## kaiko.ai - DINOv2 ViT-L14 (TCGA)

To evaluate [kaiko.ai's](https://www.kaiko.ai/) FM with the larger DINOv2 ViT-L14 backbone, pretrained on TCGA data,
run:

```
PRETRAINED=false \
EMBEDDINGS_ROOT="./data/embeddings/dinov2_vitl14_kaiko" \
REPO_OR_DIR=facebookresearch/dinov2:main \
DINO_BACKBONE=dinov2_vitl14_reg \
FORCE_RELOAD=true \
CHECKPOINT_PATH=[TBD*] \
IN_FEATURES=1024 \
NORMALIZE_MEAN=[0.5,0.5,0.5] \
NORMALIZE_STD=[0.5,0.5,0.5] \
eva predict_fit --config configs/vision/dino_vit/offline/<task>.yaml
```

\* path to public checkpoint will be added when available.


## References

 [1]: Chen: A General-Purpose Self-Supervised Model for Computational Pathology, 2023 ([arxiv](https://arxiv.org/pdf/2308.15474.pdf))