#!/bin/bash    

export MODEL_NAME=phikon_vitb16
export NORMALIZE_MEAN=[0.485, 0.456, 0.406]
export NORMALIZE_STD=[0.229, 0.224, 0.225]

source "$(dirname "$0")/../_run.sh"
run_evals