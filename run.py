import copy
import random
import importlib
import logging

import hydra
from omegaconf import OmegaConf
import numpy as np
import torch
import utils


from trainer import EditTrainer, SupervisedTrainer
import models


OmegaConf.register_new_resolver("uuid", lambda: utils.uuid())


logging.basicConfig(format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
                    level=logging.INFO)
LOG = logging.getLogger(__name__)


@hydra.main(config_path='config', config_name='config')
# load models, data, and start training
def run(config):
    LOG.info(f"\n\n{OmegaConf.to_yaml(config)}\n")
    base_dir = hydra.utils.get_original_cwd()
    LOG.info(f"Project base directory: {base_dir}")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    model = models.get_model(config) # gets pretrained model
    tokenizer = models.get_tokenizer(config)
    # inputs = tokenizer("Hello, how are you?", return_tensors="pt")
    # reply_ids = model.generate(**inputs)
    # print(tokenizer.decode(reply_ids[0], skip_special_tokens=True))
    if config.task == "qa" or config.task == "zsre":
        from data_classes.zsre import Seq2SeqAugmentedKILT

        if config.eval_only:
            train_set = val_set = Seq2SeqAugmentedKILT("test", tokenizer, config)
        else:
            train_set = Seq2SeqAugmentedKILT("train", tokenizer, config)
            val_set = Seq2SeqAugmentedKILT("dev", tokenizer, config)
    elif config.task == "sent":
        if "gpt" in model.name_or_path.lower():
            utils.add_padding(tokenizer, model)
        from data_classes.sentiment import SentimentDataset

        if config.eval_only:
            train_set = val_set = SentimentDataset(tokenizer, f"{base_dir}/data/sentiment/blender_test.json", config)
        else:
            train_set = SentimentDataset(tokenizer, f"{base_dir}/data/sentiment/blender_train.json", config)
            val_set = SentimentDataset(tokenizer, f"{base_dir}/data/sentiment/blender_val.json", config)
    elif config.task == "fnli":
        from data_classes.vitc import VitC

        if config.eval_only:
            train_set = val_set = VitC(f"{base_dir}/data/vitaminc", "test", tokenizer, config,)
        else:
            train_set = VitC(f"{base_dir}/data/vitaminc", "train", tokenizer, config)
            val_set = VitC(f"{base_dir}/data/vitaminc", "dev", tokenizer, config,)
    else:
        raise ValueError(f"Unrecognized task {config.task}")

    alg_module = importlib.import_module(f"algs.{config.alg}")
    LOG.info(f"Loading class {config.alg.upper()} from module {alg_module}")
    AlgClass = getattr(alg_module, config.alg.upper())
    alg = AlgClass(model, config, lambda: copy.deepcopy(model)) # model for editing

    if config.alg == "ft" and config.ft.locality.enabled:
        if config.ft.locality.oracle:
            alg.loc_sampler = train_set.edit_generator(config.ft.locality.batch_size + 1)
        else:
            state = np.random.get_state()
            np.random.seed(0)
            loc_batch = next(train_set.edit_generator(config.ft.locality.batch_size + 1))["loc"]
            np.random.set_state(state)
            alg.loc_ids = loc_batch["input_ids"]
            alg.loc_masks = loc_batch["attention_mask"]

    if config.playground:
        path = '/home/gridsan/shossain/serac/outputs/2024-07-02_11-48-31_3983295539/models/blenderbot_small-90M.2024-07-02_11-48-31_3983295539'
        # archive = torch.load(path, map_location="cpu")
        archive = torch.load(path, map_location="cuda:0")
        alg.cuda()
        alg.load_state_dict(archive['model'])
        alg.eval()
        
        # x = torch.arange(20).view(1, 20) + 1000
        edit_gen = train_set.edit_generator(batch_size=config.batch_size)
        x = next(edit_gen)
        # input_text = alg.replacement_tok.batch_decode(x['loc']["input_ids"], skip_special_tokens=True)
        # inputs2 = tokenizer('topic: locarno', return_tensors="pt")
        # inputs2 = {k: v.to(torch.device("cuda:0")) for k, v in inputs2.items()}
        # reply_ids = alg.generate(**inputs2)
        # orig_out = (tokenizer.decode(reply_ids[0], skip_special_tokens=True))

        orig_logits = alg(**x["loc"])
        inputs = {"input_ids": x["loc"].get("input_ids"), "attention_mask": torch.ones_like(x['loc']['input_ids'])}
        inputs = {k: v.to(torch.device("cuda:0")) for k, v in inputs.items()}
        # LEFT OFF: output text is weird; prbo from mismatch of batched inputs but not batch gen; additing batch param below doesn't work
        reply_ids = alg.generate(**inputs)
        orig_out = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
        LOG.info(f"Original output: {orig_out}")

        edited = alg.edit(x["edit_inner"])[0]
        post_logits = alg(**x["loc"])
        reply_ids = alg.generate(**inputs)
        post_out = (tokenizer.decode(reply_ids[0], skip_special_tokens=True))
        LOG.info(f"Post-edit output: {post_out}")
        assert torch.allclose(orig_logits, post_logits)

        orig_param = [p for (n, p) in alg.model.named_parameters() if n == config.model.inner_params[-1]][0]
        edited_param = [p for (n, p) in edited.model.named_parameters() if n == config.model.inner_params[-1]][0]

        LOG.info((orig_param - edited_param).abs().max())

        # edited.eval()
        # pdb.set_trace()
        # LOG.info(alg(x, labels=x).loss, edited(x, labels=x).loss, edited.edit_loss_fn(edited(x).logits, x)["nll"])
        # edited2 = alg.edit(x["edit_inner"])[0]
        # LOG.info(alg(x, labels=x).loss, edited(x, labels=x).loss, edited2(x, labels=x).loss)

        # sample = train_set.sample_completions(0)
        # import json
        # with open('data.json', 'w') as f:
        #     json.dump(sample, f, ensure_ascii=False)

        # inputs = tokenizer(" I'm not sure. ", return_tensors="pt")
        # reply_ids = alg.generate(**inputs)
        # print(tokenizer.decode(reply_ids[0], skip_special_tokens=True))

    else:
        if config.alg == "rep" and config.rep.supervised:
            trainer = SupervisedTrainer(alg, config, train_set, val_set)
        else:
            trainer = EditTrainer(alg, config, train_set, val_set)
        LOG.info(f"Built trainer: {trainer}")
        trainer.run()

    

if __name__ == "__main__":
    import pdb
    run()
