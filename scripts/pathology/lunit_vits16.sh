#!/bin/bash    

export MODEL_NAME=lunit_vits16
export NORMALIZE_MEAN=[0.70322989,0.53606487,0.66096631]
export NORMALIZE_STD=[0.21716536,0.26081574,0.20723464]

source "$(dirname "$0")/../_run.sh"
run_evals