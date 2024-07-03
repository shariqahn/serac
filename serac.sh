#!/bin/bash

# Set up correct environment
. env/bin/activate

export HYDRA_FULL_ERROR=1
# python models.py
python -m run +alg=rep +experiment=sent +model=blender-small batch_size=5 val_batch_size=5
# python -m run +alg=rep +experiment=fnli +model=bert-base batch_size=10 val_batch_size=10 rep.cross_attend=True

deactivate