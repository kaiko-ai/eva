# Replicate evaluations

To produce the evaluation results presented [here](../../index.md#evaluation-results), you can run *eva* with the settings below.

Make sure to replace `<task>` in the commands below with `bach`, `crc`, `mhist` or `patch_camelyon`.

*Note that to run the commands below you will need to first download the data. [BACH](../../datasets/bach.md), [CRC](../../datasets/crc.md) and [PatchCamelyon](../../datasets/patch_camelyon.md) provide automatic download by setting the argument `download: true` in their respective config-files. In the case of [MHIST](../../datasets/mhist.md) you will need to download the data manually by following the instructions provided [here](../../datasets/mhist.md#download-and-preprocessing).*

## DINO ViT-S16 (random weights)

Evaluating the backbone with randomly initialized weights serves as a baseline to compare the pretrained FMs 
to an FM that produces embeddings without any prior learning on image tasks. To evaluate, run:

```
# set environment variables:
export PRETRAINED=false
export EMBEDDINGS_ROOT="./data/embeddings/dino_vits16_random"
export REPO_OR_DIR=facebookresearch/dino:main
export DINO_BACKBONE=dino_vits16
export CHECKPOINT_PATH=null
export IN_FEATURES=384
export NORMALIZE_MEAN=[0.485,0.456,0.406]
export NORMALIZE_STD=[0.229,0.224,0.225]

# run eva:
eva predict_fit --config configs/vision/dino_vit/offline/<task>.yaml
```

## DINO ViT-S16 (ImageNet)

The next baseline model, uses a pretrained ViT-S16 backbone with ImageNet weights. To evaluate, run:

```
# set environment variables:
export PRETRAINED=true
export EMBEDDINGS_ROOT="./data/embeddings/dino_vits16_imagenet"
export REPO_OR_DIR=facebookresearch/dino:main
export DINO_BACKBONE=dino_vits16
export CHECKPOINT_PATH=null
export IN_FEATURES=384
export NORMALIZE_MEAN=[0.485,0.456,0.406]
export NORMALIZE_STD=[0.229,0.224,0.225]

# run eva:
eva predict_fit --config configs/vision/dino_vit/offline/<task>.yaml
```

## DINO ViT-B8 (ImageNet)

To evaluate performance on the larger ViT-B8 backbone pretrained on ImageNet, run:
```
# set environment variables:
export PRETRAINED=true
export EMBEDDINGS_ROOT="./data/embeddings/dino_vitb8_imagenet"
export REPO_OR_DIR=facebookresearch/dino:main
export DINO_BACKBONE=dino_vitb8
export CHECKPOINT_PATH=null
export IN_FEATURES=384
export NORMALIZE_MEAN=[0.485,0.456,0.406]
export NORMALIZE_STD=[0.229,0.224,0.225]

# run eva:
eva predict_fit --config configs/vision/dino_vit/offline/<task>.yaml
```

## Lunit - DINO ViT-S16 (TCGA)

[Lunit](https://www.lunit.io/en), released the weights for a DINO ViT-S16 backbone, pretrained on TCGA data
on [GitHub](https://github.com/lunit-io/benchmark-ssl-pathology/releases/). To evaluate, run:

```
# set environment variables:
export PRETRAINED=false
export EMBEDDINGS_ROOT="./data/embeddings/dino_vits16_lunit"
export REPO_OR_DIR=facebookresearch/dino:main
export DINO_BACKBONE=dino_vits16
export CHECKPOINT_PATH="https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights/dino_vit_small_patch16_ep200.torch"
export IN_FEATURES=384
export NORMALIZE_MEAN=[0.70322989,0.53606487,0.66096631]
export NORMALIZE_STD=[0.21716536,0.26081574,0.20723464]

# run eva:
eva predict_fit --config configs/vision/dino_vit/offline/<task>.yaml
```

## Owkin - iBOT ViT-B16 (TCGA)

[Owkin](https://www.owkin.com/) released the weights for "Phikon", an FM trained with iBOT on TCGA data, via
[HuggingFace](https://huggingface.co/owkin/phikon). To evaluate, run:

```
# set environment variables:
export EMBEDDINGS_ROOT="./data/embeddings/dino_vitb16_owkin"

# run eva:
eva predict_fit --config configs/vision/owkin/phikon/offline/<task>.yaml
```

Note: since ***eva*** provides the config files to evaluate tasks with the Phikon FM in 
"configs/vision/owkin/phikon/offline", it is not necessary to set the environment variables needed for
the runs above.

## kaiko.ai - DINO ViT-S16 (TCGA)

To evaluate [kaiko.ai's](https://www.kaiko.ai/) FM with DINO ViT-S16 backbone, pretrained on TCGA data 
on [GitHub](https://github.com/lunit-io/benchmark-ssl-pathology/releases/), run:

```
# set environment variables:
export PRETRAINED=false
export EMBEDDINGS_ROOT="./data/embeddings/dino_vits16_kaiko"
export REPO_OR_DIR=facebookresearch/dino:main
export DINO_BACKBONE=dino_vits16
export CHECKPOINT_PATH=[TBD*]
export IN_FEATURES=384
export NORMALIZE_MEAN=[0.5,0.5,0.5]
export NORMALIZE_STD=[0.5,0.5,0.5]

# run eva:
eva predict_fit --config configs/vision/dino_vit/offline/<task>.yaml
```

\* path to public checkpoint will be added when available.

## kaiko.ai - DINO ViT-S8 (TCGA)

To evaluate [kaiko.ai's](https://www.kaiko.ai/) FM with DINO ViT-S8 backbone, pretrained on TCGA data 
on [GitHub](https://github.com/lunit-io/benchmark-ssl-pathology/releases/), run:

```
# set environment variables:
export PRETRAINED=false
export EMBEDDINGS_ROOT="./data/embeddings/dino_vits8_kaiko"
export REPO_OR_DIR=facebookresearch/dino:main
export DINO_BACKBONE=dino_vits8
export CHECKPOINT_PATH=[TBD*]
export IN_FEATURES=384
export NORMALIZE_MEAN=[0.5,0.5,0.5]
export NORMALIZE_STD=[0.5,0.5,0.5]

# run eva:
eva predict_fit --config configs/vision/dino_vit/offline/<task>.yaml
```

\* path to public checkpoint will be added when available.

## kaiko.ai - DINO ViT-B16 (TCGA)

To evaluate [kaiko.ai's](https://www.kaiko.ai/) FM with the larger DINO ViT-B16 backbone, pretrained on TCGA data,
run:

```
# set environment variables:
export PRETRAINED=false
export EMBEDDINGS_ROOT="./data/embeddings/dino_vitb16_kaiko"
export REPO_OR_DIR=facebookresearch/dino:main
export DINO_BACKBONE=dino_vitb16
export CHECKPOINT_PATH=[TBD*]
export IN_FEATURES=768
export NORMALIZE_MEAN=[0.5,0.5,0.5]
export NORMALIZE_STD=[0.5,0.5,0.5]

# run eva:
eva predict_fit --config configs/vision/dino_vit/offline/<task>.yaml
```

\* path to public checkpoint will be added when available.

## kaiko.ai - DINO ViT-B8 (TCGA)

To evaluate [kaiko.ai's](https://www.kaiko.ai/) FM with the larger DINO ViT-B8 backbone, pretrained on TCGA data,
run:

```
# set environment variables:
export PRETRAINED=false
export EMBEDDINGS_ROOT="./data/embeddings/dino_vitb8_kaiko"
export REPO_OR_DIR=facebookresearch/dino:main
export DINO_BACKBONE=dino_vitb8
export CHECKPOINT_PATH=[TBD*]
export IN_FEATURES=768
export NORMALIZE_MEAN=[0.5,0.5,0.5]
export NORMALIZE_STD=[0.5,0.5,0.5]

# run eva:
eva predict_fit --config configs/vision/dino_vit/offline/<task>.yaml
```

\* path to public checkpoint will be added when available.

## kaiko.ai - DINOv2 ViT-L14 (TCGA)

To evaluate [kaiko.ai's](https://www.kaiko.ai/) FM with the larger DINOv2 ViT-L14 backbone, pretrained on TCGA data,
run:

```
# set environment variables:
export PRETRAINED=false
export EMBEDDINGS_ROOT="./data/embeddings/dinov2_vitl14_kaiko"
export REPO_OR_DIR=facebookresearch/dinov2:main
export DINO_BACKBONE=dinov2_vitl14_reg
export FORCE_RELOAD=true
export CHECKPOINT_PATH=[TBD*]
export IN_FEATURES=1024
export NORMALIZE_MEAN=[0.5,0.5,0.5]
export NORMALIZE_STD=[0.5,0.5,0.5]

# run eva:
eva predict_fit --config configs/vision/dino_vit/offline/<task>.yaml
```

\* path to public checkpoint will be added when available.