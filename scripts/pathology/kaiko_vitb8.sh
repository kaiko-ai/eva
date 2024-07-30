#!/bin/bash    

export MODEL_NAME=kaiko_vitb8
export NORMALIZE_MEAN=[0.5,0.5,0.5]
export NORMALIZE_STD=[0.5,0.5,0.5]

source "$(dirname "$0")/../_run.sh"
run_evals