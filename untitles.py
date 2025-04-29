from huggingface_hub import snapshot_download
import transformers
from easyeditor import MENDTrainingHparams

# snapshot_download(repo_id=config.model.name, cache_dir=cache_dir)
# snapshot_download(repo_id='sentence-transformers/all-MiniLM-L6-v2')

# zsre
def get_zsre():
    tok_name = (
        config.tokenizer_name
        if config.tokenizer_name is not None
        else config.model.name
    )
    tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
        tok_name, trust_remote_code=True
    )

if __name__ == "__main__":
    config = MENDTrainingHparams.from_hparams('hparams/TRAINING/MEND/llama-7b.yaml')
    get_zsre(config)