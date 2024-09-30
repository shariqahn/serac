#!/bin/bash

# Set up correct environment
. env/bin/activate
export HYDRA_FULL_ERROR=1
python -m run +alg=rep +experiment=qa +model=t5large batch_size=10 val_batch_size=10 data.zsre_impl=true data.zsre_yn=true data.hard_neg=true
deactivate

# python -m run +alg=rep +experiment=sent +model=blender-small batch_size=5 val_batch_size=5 
# python -m run +alg=rep +experiment=sent +model=blender-small batch_size=5 val_batch_size=5 playground=True