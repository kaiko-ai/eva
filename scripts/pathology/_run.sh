#!/bin/bash    

CONFIGS=(
        "bach.yaml"
        "bcss.yaml"
        "camelyon16.yaml"
        "consep.yaml"
        "crc.yaml"
        "mhist.yaml"
        "monusac.yaml"
        "panda.yaml"
        "patch_camelyon.yaml"
)

run_evals() {
    for config in "${CONFIGS[@]}"; do
        echo "Running $config ..."
        eva predict_fit --config configs/vision/pathology/offline/${config}
    done
}