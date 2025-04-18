# Replicate evaluations

To produce the evaluation results presented [here](../../leaderboards.md), you can run *eva* with the settings below.

The `.yaml` config files for the different benchmark datasets can be found on [GitHub](https://github.com/kaiko-ai/eva/tree/main/configs/vision).
You will need to download the config files and then in the following commands replace `<task.yaml>` with the name of the config you want to use.

Keep in mind:

- Some datasets provide automatic download by setting the argument `download: true` (either modify the `.yaml` config file or set the environment variable `DOWNLOAD=true`), while other datasets need to be downloaded manually beforehand. Please review the instructions in the corresponding dataset [documentation](../../datasets/index.md).
- The following `eva predict_fit` commands will store the generated embeddings to the `./data/embeddings` directory. To change this location you can alternatively set the `EMBEDDINGS_ROOT` environment variable.
- Segmentation tasks need to be run in `online` mode because the decoder currently doesn't support evaluation with precomputed embeddings. In other words, use `fit --config .../online/<task>.yaml` instead of `predict_fit  --config .../offline/<task>.yam` here.


## Pathology FMs

### DINO ViT-S16 (random weights)

Evaluating the backbone with randomly initialized weights serves as a baseline to compare the pretrained FMs to a FM that produces embeddings without any prior learning on image tasks. To evaluate, run:

```
MODEL_NAME="universal/vit_small_patch16_224_random" \
NORMALIZE_MEAN="[0.485,0.456,0.406]" \
NORMALIZE_STD="[0.229,0.224,0.225]" \
IN_FEATURES=384 \
eva predict_fit --config configs/vision/pathology/offline/<task>.yaml
```

### DINO ViT-S16 (ImageNet)

The next baseline model, uses a pretrained ViT-S16 backbone with ImageNet weights. To evaluate, run:

```
MODEL_NAME="universal/vit_small_patch16_224_dino" \
NORMALIZE_MEAN="[0.485,0.456,0.406]" \
NORMALIZE_STD="[0.229,0.224,0.225]" \
IN_FEATURES=384 \
eva predict_fit --config configs/vision/pathology/offline/<task>.yaml
```

### Lunit - DINO ViT-S16 (TCGA) [[1]](#references)

[Lunit](https://www.lunit.io/en), released the weights for a DINO ViT-S16 backbone, pretrained on TCGA data
on [GitHub](https://github.com/lunit-io/benchmark-ssl-pathology/releases/). To evaluate, run:

```
MODEL_NAME=pathology/lunit_vits16
NORMALIZE_MEAN="[0.70322989,0.53606487,0.66096631]" \
NORMALIZE_STD="[0.21716536,0.26081574,0.20723464]" \
IN_FEATURES=384 \
eva predict_fit --config configs/vision/pathology/offline/<task>.yaml
```

### Lunit - DINO ViT-S8 (TCGA) [[1]](#references)

```
MODEL_NAME=pathology/lunit_vits8 \
NORMALIZE_MEAN="[0.70322989,0.53606487,0.66096631]" \
NORMALIZE_STD="[0.21716536,0.26081574,0.20723464]" \
IN_FEATURES=384 \
eva predict_fit --config configs/vision/pathology/offline/<task>.yaml
```

### Phikon (Owkin) - iBOT ViT-B16 (TCGA) [[2]](#references)

[Owkin](https://www.owkin.com/) released the weights for "Phikon", a FM trained with iBOT on TCGA data, via
[HuggingFace](https://huggingface.co/owkin/phikon). To evaluate, run:

```
MODEL_NAME=pathology/owkin_phikon \
NORMALIZE_MEAN="[0.485,0.456,0.406]" \
NORMALIZE_STD="[0.229,0.224,0.225]" \
IN_FEATURES=768 \
eva predict_fit --config configs/vision/pathology/offline/<task>.yaml
```

### Phikon-v2 (Owkin) - DINOv2 ViT-L16 (PANCAN-XL) [[9]](#references)

[Owkin](https://www.owkin.com/) released the weights for "Phikon-v2", a FM trained with DINOv2
on the PANCAN-XL dataset (450M 20x magnification histology images sampled from 60K WSIs), via
[HuggingFace](https://huggingface.co/owkin/phikon-v2). To evaluate, run:

```
MODEL_NAME=pathology/owkin_phikon_v2 \
NORMALIZE_MEAN="[0.485,0.456,0.406]" \
NORMALIZE_STD="[0.229,0.224,0.225]" \
IN_FEATURES=1024 \
eva predict_fit --config configs/vision/pathology/offline/<task>.yaml
```

### UNI (MahmoodLab) - DINOv2 ViT-L16 (Mass-100k) [[3]](#references)

The UNI FM by MahmoodLab is available on [HuggingFace](https://huggingface.co/MahmoodLab/UNI). Note that access needs to 
be requested.

```
MODEL_NAME=pathology/mahmood_uni \
NORMALIZE_MEAN="[0.485,0.456,0.406]" \
NORMALIZE_STD="[0.229,0.224,0.225]" \
IN_FEATURES=1024 \
HF_TOKEN=<your-huggingace-token-for-downloading-the-model> \
eva predict_fit --config configs/vision/phikon/offline/<task>.yaml
```

### UNI2-h (MahmoodLab) - DINOv2 ViT-G14 [[3]](#references)

The UNI2-h FM by MahmoodLab is available on [HuggingFace](https://huggingface.co/MahmoodLab/UNI). Note that access needs to 
be requested.

```
MODEL_NAME=pathology/mahmood_uni2_h \
NORMALIZE_MEAN="[0.485,0.456,0.406]" \
NORMALIZE_STD="[0.229,0.224,0.225]" \
IN_FEATURES=1536 \
HF_TOKEN=<your-huggingace-token-for-downloading-the-model> \
eva predict_fit --config configs/vision/phikon/offline/<task>.yaml
```

### kaiko.ai - DINO ViT-S16 (TCGA) [[4]](#references)

To evaluate [kaiko.ai's](https://www.kaiko.ai/) FM with DINO ViT-S16 backbone, pretrained on TCGA data 
and available on [GitHub](https://github.com/kaiko-ai/towards_large_pathology_fms), run:

```
MODEL_NAME=pathology/kaiko_vits16 \
NORMALIZE_MEAN="[0.5,0.5,0.5]" \
NORMALIZE_STD="[0.5,0.5,0.5]" \
IN_FEATURES=384 \
eva predict_fit --config configs/vision/pathology/offline/<task>.yaml
```

### kaiko.ai - DINO ViT-S8 (TCGA) [[4]](#references)

To evaluate [kaiko.ai's](https://www.kaiko.ai/) FM with DINO ViT-S8 backbone, pretrained on TCGA data 
and available on [GitHub](https://github.com/kaiko-ai/towards_large_pathology_fms), run:

```
MODEL_NAME=pathology/kaiko_vits8 \
NORMALIZE_MEAN="[0.5,0.5,0.5]" \
NORMALIZE_STD="[0.5,0.5,0.5]" \
IN_FEATURES=384 \
eva predict_fit --config configs/vision/pathology/offline/<task>.yaml
```

### kaiko.ai - DINO ViT-B16 (TCGA) [[4]](#references)

To evaluate [kaiko.ai's](https://www.kaiko.ai/) FM with DINO ViT-B16 backbone, pretrained on TCGA data 
and available on [GitHub](https://github.com/kaiko-ai/towards_large_pathology_fms), run:

```
MODEL_NAME=pathology/kaiko_vitb16 \
NORMALIZE_MEAN="[0.5,0.5,0.5]" \
NORMALIZE_STD="[0.5,0.5,0.5]" \
IN_FEATURES=768 \
eva predict_fit --config configs/vision/pathology/offline/<task>.yaml
```

### kaiko.ai - DINO ViT-B8 (TCGA) [[4]](#references)

To evaluate [kaiko.ai's](https://www.kaiko.ai/) FM with DINO ViT-B8 backbone, pretrained on TCGA data 
and available on [GitHub](https://github.com/kaiko-ai/towards_large_pathology_fms), run:

```
MODEL_NAME=pathology/kaiko_vitb8 \
NORMALIZE_MEAN="[0.5,0.5,0.5]" \
NORMALIZE_STD="[0.5,0.5,0.5]" \
IN_FEATURES=768 \
eva predict_fit --config configs/vision/pathology/offline/<task>.yaml
```

### kaiko.ai - DINOv2 ViT-L14 (TCGA) [[4]](#references)

To evaluate [kaiko.ai's](https://www.kaiko.ai/) FM with DINOv2 ViT-L14 backbone, pretrained on TCGA data 
and available on [GitHub](https://github.com/kaiko-ai/towards_large_pathology_fms), run:

```
MODEL_NAME=pathology/kaiko_vitl14 \
NORMALIZE_MEAN="[0.5,0.5,0.5]" \
NORMALIZE_STD="[0.5,0.5,0.5]" \
IN_FEATURES=1024 \
eva predict_fit --config configs/vision/pathology/offline/<task>.yaml
```

### kaiko.ai - DINOv2 Midnight-12k (TCGA) [[4]](#references)

To evaluate [kaiko.ai's](https://www.kaiko.ai/) FM with Midnight-12k (ViT-G14) backbone, pretrained on TCGA data 
and available on [GitHub](https://github.com/kaiko-ai/Midnight), run:

```
MODEL_NAME=pathology/kaiko_midnight_12k \
NORMALIZE_MEAN="[0.5,0.5,0.5]" \
NORMALIZE_STD="[0.5,0.5,0.5]" \
IN_FEATURES=1536 \
eva predict_fit --config configs/vision/pathology/offline/<task>.yaml
```


### H-optimus-0 (Bioptimus) - ViT-G14 [[5]](#references)
[Bioptimus](https://www.bioptimus.com) released their H-optimus-0 which was trained on a collection of 500,000 H&E slides. The model weights
were released on [HuggingFace](https://huggingface.co/bioptimus/H-optimus-0).

```
MODEL_NAME=pathology/bioptimus_h_optimus_0 \
NORMALIZE_MEAN="[0.707223,0.578729,0.703617]" \
NORMALIZE_STD="[0.211883,0.230117,0.177517]" \
IN_FEATURES=1536 \
eva predict_fit --config configs/vision/pathology/offline/<task>.yaml
```


### Prov-GigaPath - DINOv2 ViT-G14 [[6]](#references)
To evaluate the [Prov-Gigapath](https://github.com/prov-gigapath/prov-gigapath) model, available on [HuggingFace](https://huggingface.co/prov-gigapath/prov-gigapath), run:

```
MODEL_NAME=pathology/prov_gigapath \
NORMALIZE_MEAN="[0.485,0.456,0.406]" \
NORMALIZE_STD="[0.229,0.224,0.225]" \
IN_FEATURES=1536 \
eva predict_fit --config configs/vision/pathology/offline/<task>.yaml
```


### hibou-B (hist.ai) - DINOv2 ViT-B14 (1M Slides) [[7]](#references)
To evaluate [hist.ai's](https://www.hist.ai/) FM with DINOv2 ViT-B14 backbone, pretrained on
a proprietary dataset of one million slides, available for download on
[HuggingFace](https://huggingface.co/histai/hibou-b), run: 

```
MODEL_NAME=pathology/histai_hibou_b \
NORMALIZE_MEAN="[0.7068,0.5755,0.722]" \
NORMALIZE_STD="[0.195,0.2316,0.1816]" \
IN_FEATURES=768 \
eva predict_fit --config configs/vision/pathology/offline/<task>.yaml
```

### hibou-L (hist.ai) - DINOv2 ViT-L14 (1M Slides) [[7]](#references)
To evaluate [hist.ai's](https://www.hist.ai/) FM with DINOv2 ViT-L14 backbone, pretrained on
a proprietary dataset of one million slides, available for download on
[HuggingFace](https://huggingface.co/histai/hibou-l), run: 

```
MODEL_NAME=pathology/histai_hibou_l \
NORMALIZE_MEAN="[0.7068,0.5755,0.722]" \
NORMALIZE_STD="[0.195,0.2316,0.1816]" \
IN_FEATURES=1024 \
eva predict_fit --config configs/vision/pathology/offline/<task>.yaml
```

### Virchow2 (paige.ai) - DINOv2 ViT-H14 (3.1M Slides) [[8]](#references)
To evaluate [paige.ai's](https://www.paige.ai/) FM with DINOv2 ViT-H14 backbone, pretrained on
a proprietary dataset of 3.1M million slides, available for download on
[HuggingFace](https://huggingface.co/paige-ai/Virchow2), run:

```
MODEL_NAME=pathology/paige_virchow2 \
NORMALIZE_MEAN="[0.485,0.456,0.406]" \
NORMALIZE_STD="[0.229,0.224,0.225]" \
IN_FEATURES=1280 \
eva predict_fit --config configs/vision/pathology/offline/<task>.yaml
```


## References

 [1]: Kang, Mingu, et al. "Benchmarking self-supervised learning on diverse pathology datasets." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

 [2]: Filiot, Alexandre, et al. "Scaling self-supervised learning for histopathology with masked image modeling." medRxiv (2023): 2023-07.
 
 [3]: Chen: Chen, Richard J., et al. "A general-purpose self-supervised model for computational pathology." arXiv preprint arXiv:2308.15474 (2023).

 [4]: Aben, Nanne, et al. "Towards Large-Scale Training of Pathology Foundation Models." arXiv preprint arXiv:2404.15217 (2024).

 [5]: Saillard, et al. "H-optimus-0" https://github.com/bioptimus/releases/tree/main/models/h-optimus/v0 (2024).

 [6]: Xu, Hanwen, et al. "A whole-slide foundation model for digital pathology from real-world data." Nature (2024): 1-8.

 [7]: Nechaev, Dmitry, Alexey Pchelnikov, and Ekaterina Ivanova. "Hibou: A Family of Foundational Vision Transformers for Pathology." arXiv preprint arXiv:2406.05074 (2024).

 [8]: Zimmermann, Eric, et al. "Virchow 2: Scaling Self-Supervised Mixed Magnification Models in Pathology." arXiv preprint arXiv:2408.00738 (2024).

 [9]: Filiot, Alexandre, et al. "Phikon-v2, A large and public feature extractor for biomarker prediction." arXiv preprint arXiv:2409.09173 (2024).

 [10]: Chen, Richard J., et al. "Towards a general-purpose foundation model for computational pathology." Nature Medicine 30.3 (2024): 850-862.

 [11]: Karasikov, Mikhail, et al. "Training state-of-the-art pathology foundation models with orders of magnitude less data" arXiv preprint arXiv:2504.05186850-862.