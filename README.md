## Semi-Parametric Editing with a Retrieval-Augmented Counterfactual Model

Code and data for the ICML 2022 paper *Memory-based Model Editing at Scale*.

See the paper [here](https://arxiv.org/pdf/2206.06520.pdf) and the project website [here](https://sites.google.com/view/serac-editing).

## Setup

### Environment

This codebase uses Python 3.7.9. Other versions may work as well.

Create a virtualenv ([pyenv](https://github.com/pyenv/pyenv) can help with this)
and install the dependencies:

    $ python -m venv env
    $ source env/bin/activate
    (env) $ pip install -r requirements.txt

### Data

You can download the data needed for this project from
[this Google Drive link](https://drive.google.com/file/d/1W-7Yb0eMxwZqdr7aeSgvZnbFKkzwavn6/view?usp=sharing).
You just need to unzip the archive into the top-level `serac` directory.

## Running the code

You can run the code with:

    (env) $ python -m run +alg=ALG +experiment=EXP +model=MODEL
    
See the `scripts/` directory for examples. `ALG` may be one of:
- rep [SERAC]
- gtn [MEND; [Mitchell et al., 2022](https://arxiv.org/pdf/2110.11309.pdf)]
- enn [Editable Neural Networks; [Sinitsin et al., 2020](https://arxiv.org/pdf/2004.00345.pdf)]
- lu [lookup cache baseline]
- ft [fine-tuning baseline]

The `EXP` argument may be one of:
- zsre [question-answering; must be used with `MODEL=t5large`]
- fnli [fact-checking; must be used with `MODEL=bert-base`]
- sent [sentiment editing; must be used with `MODEL=blender-small`]

## Citing the paper
If this repository is useful for your own research, you can cite our work with the following BibTeX entry:

    @inproceedings{mitchell2022memory,
        title={Memory-Based Model Editing at Scale},
        author={Mitchell, Eric and Lin, Charles and Bosselut, Antoine and Finn, Chelsea and Manning, Christopher D.},
        booktitle={International Conference on Machine Learning},
        url={https://arxiv.org/pdf/2206.06520.pdf},
        year={2022},
    }  

# My Notes
- set config params like batch is done here: 
`python -m run +alg=rep +experiment=sent +model=blender-small batch_size=5 val_batch_size=5`

## Models
- fb blenderbot: for chatbots
- bert: masked language modeling (MLM), next sentence prediction; usually used for fine-tuning

- dataset: "ent" = "entity" = topic were evaluating sentiment on

- run training script: 
    - batch: `LLsub serac.sh -g volta:1`
        - with cpus: `LLsub serac.sh -s 1 -g volta:1`
    - serial: 
        ```
        LLsub -i -g volta:1
        conda deactivate
        conda activate cenv
        export HYDRA_FULL_ERROR=1
        python -m run +alg=rep +experiment=qa +model=t5large batch_size=10 val_batch_size=10 data.zsre_impl=true data.zsre_yn=true data.hard_neg=true
        ```
        - download: `LLsub -i -q download`

<!-- left off comment is where i left off hehe -->

## Set Up Log
- use conda so sqlite dependency is set up properly
- match python version of serac: 3.7.9
- upgrade pip to the latest version to ensure dependencies are compatible and resolved in an efficient manner
- install dependencies. This takes a long time because pip has to resolve incompatibilities
- download models in specific state... dir
    - open download partition
    - will automatically download in state... dir (?)
    - copy to local
    - running download_models.sh in serac dir will delete existing downloaded models and download the necessary ones for the selected experiment config