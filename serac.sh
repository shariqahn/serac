#!/bin/bash

# Set up correct environment
# export CUDA_LAUNCH_BLOCKING=1
# export HYDRA_FULL_ERROR=1

source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023b/etc/profile.d/conda.sh
conda deactivate 
conda activate cenv

# python -m run +alg=rep +experiment=qa +model=t5large batch_size=10 val_batch_size=10 data.zsre_impl=true data.zsre_yn=true data.hard_neg=true early_stop_patience=1000 eval_only=true

# python -m run +alg=rep +experiment=qa +model=t5large batch_size=10 val_batch_size=10 data.zsre_impl=true data.zsre_yn=true data.hard_neg=true

python -m run +alg=gtn +experiment=qa +model=t5large batch_size=10 val_batch_size=10 data.zsre_impl=true data.zsre_yn=true data.hard_neg=true gtn.descent=True

# python -m run +alg=rep +experiment=sent +model=blender-small batch_size=5 val_batch_size=5 
# python -m run +alg=rep +experiment=sent +model=blender-small batch_size=5 val_batch_size=5 playground=True
conda deactivate