#!/bin/bash

# Set up correct environment
. env/bin/activate

export HYDRA_FULL_ERROR=1
# this is where model is downloaded
export HF_HOME=/state/partition1/user/$USER/hug
mkdir -p $HF_HOME
HF_LOCAL_DIR=$HOME/serac/scr/shossain
mkdir -p $HF_LOCAL_DIR

echo "Dirs created:"
ls /state/partition1/user/$USER
# ls $HOME/serac
# cd $HF_HOME
# Run the script
# creates scp in dir that script is run

# sent:
# python -m collect_models +alg=rep +experiment=sent +model=blender-small batch_size=5 val_batch_size=5
# qa (not qa-hard i think?):
python -m collect_models +alg=rep +experiment=qa +model=t5large batch_size=10 val_batch_size=10 data.zsre_impl=true data.zsre_yn=true data.hard_neg=true

# Copy the model from HF_HOME into HF_LOCAL_DIR
echo "Model collected. Here is what home looks like:"
ls $HF_HOME
# ls $HF_HOME/models--facebook--blenderbot_small-90M/snapshots
cp -rf $HF_HOME/* $HF_LOCAL_DIR
echo "Model copied. Here is what local looks like:"
ls $HF_LOCAL_DIR
rm -rf $HF_HOME
echo "Home cleared. Here is what home looks like:"
ls /state/partition1/user/$USER

deactivate