#!/bin/sh
export OUTPUT_DIR="logs/dino_vits16/patch_camelyon"
export OUT_FEATURES=384
export NORMALIZE_MEAN="[0.485, 0.456, 0.406]"
export NORMALIZE_STD="[0.229, 0.224, 0.225]"

pdm run eva fit \
    --config configs/vision/tasks/patch_camelyon.yaml \
    --config configs/vision/models/dino_vits16.yaml
