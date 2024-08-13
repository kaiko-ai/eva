# Replicate evaluations

To produce the evaluation results presented [here](../../leaderboards.md), you can run *eva* with the settings below.

The `.yaml` config files for the different benchmark datasets can be found on [GitHub](https://github.com/kaiko-ai/eva/tree/main/configs/vision).
You will need to download the config files and then in the following commands replace `<task.yaml>` with the name of the config you want to use.

Keep in mind:

- Some datasets provide automatic download by setting the argument `download: true` (either modify the `.yaml` config file or set the environment variable `DOWNLOAD=true`), while other datasets need to be downloaded manually beforehand. Please review the instructions in the corresponding dataset [documentation](../../datasets/index.md).
- The following `eva predict_fit` commands will store the generated embeddings to the `./data/embeddings` directory. To change this location you can alternatively set the `EMBEDDINGS_ROOT` environment variable.


## Pathology FMs

### DINO ViT-S16 (random weights)

Evaluating the backbone with randomly initialized weights serves as a baseline to compare the pretrained FMs to an FM that produces embeddings without any prior learning on image tasks. To evaluate, run:

```
MODEL_NAME="universal/vit_small_patch16_224_random" \
eva predict_fit --config configs/vision/pathology/offline/<task>.yaml
```

### DINO ViT-S16 (ImageNet)

The next baseline model, uses a pretrained ViT-S16 backbone with ImageNet weights. To evaluate, run:

```
MODEL_NAME="universal/vit_small_patch16_224_imagenet" \
eva predict_fit --config configs/vision/pathology/offline/<task>.yaml
```

### Lunit - DINO ViT-S16 (TCGA) [[1]](#references)

[Lunit](https://www.lunit.io/en), released the weights for a DINO ViT-S16 backbone, pretrained on TCGA data
on [GitHub](https://github.com/lunit-io/benchmark-ssl-pathology/releases/). To evaluate, run:

```
MODEL_NAME=pathology/lunit_vits16
NORMALIZE_MEAN=[0.70322989,0.53606487,0.66096631] \
NORMALIZE_STD=[0.21716536,0.26081574,0.20723464] \
eva predict_fit --config configs/vision/pathology/offline/<task>.yaml
```

### Lunit - DINO ViT-S8 (TCGA) [[1]](#references)

```
MODEL_NAME=pathology/lunit_vits8 \
NORMALIZE_MEAN=[0.70322989,0.53606487,0.66096631] \
NORMALIZE_STD=[0.21716536,0.26081574,0.20723464] \
eva predict_fit --config configs/vision/pathology/offline/<task>.yaml
```

### Phikon (Owkin) - iBOT ViT-B16 (TCGA) [[2]](#references)

[Owkin](https://www.owkin.com/) released the weights for "Phikon", an FM trained with iBOT on TCGA data, via
[HuggingFace](https://huggingface.co/owkin/phikon). To evaluate, run:

```
MODEL_NAME=pathology/owkin_phikon \
eva predict_fit --config configs/vision/pathology/offline/<task>.yaml
```

### UNI - DINOv2 ViT-L16 (Mass-100k) [[3]](#references)

The UNI FM by MahmoodLab is available on [HuggingFace](https://huggingface.co/MahmoodLab/UNI). Note that access needs to 
be requested.

```
MODEL_NAME=pathology/mahmood_uni \
HF_TOKEN=<your-huggingace-token-for-downloading-the-model> \
IN_FEATURES=1024 \
eva predict_fit --config configs/vision/phikon/offline/<task>.yaml
```


### kaiko.ai - DINO ViT-S16 (TCGA) [[4]](#references)

To evaluate [kaiko.ai's](https://www.kaiko.ai/) FM with DINO ViT-S16 backbone, pretrained on TCGA data 
and available on [GitHub](https://github.com/kaiko-ai/towards_large_pathology_fms), run:

```
MODEL_NAME=pathology/kaiko_vits16 \
NORMALIZE_MEAN=[0.5,0.5,0.5] \
NORMALIZE_STD=[0.5,0.5,0.5] \
eva predict_fit --config configs/vision/pathology/offline/<task>.yaml
```

### kaiko.ai - DINO ViT-S8 (TCGA) [[4]](#references)

To evaluate [kaiko.ai's](https://www.kaiko.ai/) FM with DINO ViT-S8 backbone, pretrained on TCGA data 
and available on [GitHub](https://github.com/kaiko-ai/towards_large_pathology_fms), run:

```
MODEL_NAME=pathology/kaiko_vits8 \
NORMALIZE_MEAN=[0.5,0.5,0.5] \
NORMALIZE_STD=[0.5,0.5,0.5] \
eva predict_fit --config configs/vision/pathology/offline/<task>.yaml
```


### kaiko.ai - DINO ViT-B16 (TCGA) [[4]](#references)

To evaluate [kaiko.ai's](https://www.kaiko.ai/) FM with DINO ViT-B16 backbone, pretrained on TCGA data 
and available on [GitHub](https://github.com/kaiko-ai/towards_large_pathology_fms), run:

```
MODEL_NAME=pathology/kaiko_vitb16 \
NORMALIZE_MEAN=[0.5,0.5,0.5] \
NORMALIZE_STD=[0.5,0.5,0.5] \
IN_FEATURES=768 \
eva predict_fit --config configs/vision/pathology/offline/<task>.yaml
```

### kaiko.ai - DINO ViT-B8 (TCGA) [[4]](#references)

To evaluate [kaiko.ai's](https://www.kaiko.ai/) FM with DINO ViT-B8 backbone, pretrained on TCGA data 
and available on [GitHub](https://github.com/kaiko-ai/towards_large_pathology_fms), run:

```
MODEL_NAME=pathology/kaiko_vitb16 \
NORMALIZE_MEAN=[0.5,0.5,0.5] \
NORMALIZE_STD=[0.5,0.5,0.5] \
IN_FEATURES=768 \
eva predict_fit --config configs/vision/pathology/offline/<task>.yaml
```


### kaiko.ai - DINOv2 ViT-L14 (TCGA) [[4]](#references)

To evaluate [kaiko.ai's](https://www.kaiko.ai/) FM with DINOv2 ViT-L14 backbone, pretrained on TCGA data 
and available on [GitHub](https://github.com/kaiko-ai/towards_large_pathology_fms), run:

```
MODEL_NAME=pathology/kaiko_vitl14 \
NORMALIZE_MEAN=[0.5,0.5,0.5] \
NORMALIZE_STD=[0.5,0.5,0.5] \
IN_FEATURES=1024 \
eva predict_fit --config configs/vision/pathology/offline/<task>.yaml
```

### hibou-B (hist.ai) - DINOv2 ViT-B14 (1M Slides) [[7]](#references)
To evaluate [hist.ai's](https://www.hist.ai/) FM with DINOv2 ViT-B14 backbone, pretrained on
a proprietary dataset of one million slides, available for download on
[HuggingFace](https://huggingface.co/histai/hibou-b), run: 

```
MODEL_NAME=pathology/histai_hibou_b \
NORMALIZE_MEAN=[0.7068,0.5755,0.722] \
NORMALIZE_STD=[0.195,0.2316,0.1816] \
IN_FEATURES=768 \
eva predict_fit --config configs/vision/pathology/offline/<task>.yaml
```

### hibou-L (hist.ai) - DINOv2 ViT-L14 (1M Slides) [[7]](#references)
To evaluate [hist.ai's](https://www.hist.ai/) FM with DINOv2 ViT-L14 backbone, pretrained on
a proprietary dataset of one million slides, available for download on
[HuggingFace](https://huggingface.co/histai/hibou-l), run: 

```
MODEL_NAME=pathology/histai_hibou_l \
NORMALIZE_MEAN=[0.7068,0.5755,0.722] \
NORMALIZE_STD=[0.195,0.2316,0.1816] \
IN_FEATURES=1024 \
eva predict_fit --config configs/vision/pathology/offline/<task>.yaml
```

## References

 [1]: Kang, Mingu, et al. "Benchmarking self-supervised learning on diverse pathology datasets." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

 [2]: Filiot, Alexandre, et al. "Scaling self-supervised learning for histopathology with masked image modeling." medRxiv (2023): 2023-07.
 
 [3]: Chen: Chen, Richard J., et al. "A general-purpose self-supervised model for computational pathology." arXiv preprint arXiv:2308.15474 (2023).

 [4]: Aben, Nanne, et al. "Towards Large-Scale Training of Pathology Foundation Models." arXiv preprint arXiv:2404.15217 (2024).

 [7]: Nechaev, Dmitry, Alexey Pchelnikov, and Ekaterina Ivanova. "Hibou: A Family of Foundational Vision Transformers for Pathology." arXiv preprint arXiv:2406.05074 (2024).