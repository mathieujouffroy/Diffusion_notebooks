import os
import yaml
import wandb
import random
import logging
import argparse
import datetime
import numpy as np
import torch


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def set_logging(args, log_dir):
    "Defines the file in which we will write our training logs"
    
    date = datetime.datetime.now().strftime("%d:%m-%H:%M")
    log_file = os.path.join(log_dir, f"log_{date}.log")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
        #force=True
    )
    

class YamlNamespace(argparse.Namespace):
    """Namespace from a nested dict returned by yaml.load()"""

    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [YamlNamespace(x)
                        if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, YamlNamespace(b)
                        if isinstance(b, dict) else b)


def wandb_cfg(args):
    # SETUP WANDB
    config_dict = {
        "dataset_name": args.dataset_name,
        "model_name": args.output_dir,
        "seed": args.seed,
        "resolution": args.resolution,
        "nbr_channels": args.nbr_channels,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_epochs": args.num_epochs,
        "pretrained_model": args.pretrained_model,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "lr_scheduler": args.lr_scheduler,
        "lr_warmup_steps": args.lr_warmup_steps,
        "adam_weight_decay": args.adam_weight_decay,
        "ddpm_num_steps": args.ddpm_num_steps,
        "ddpm_beta_schedule": args.ddpm_beta_schedule,
        "loss_type": args.loss_type,
        "use_ema": args.use_ema,
    }
    return config_dict


def set_wandb_project_run(args):
    """ Initialize wandb directory to keep track of our models. """

    cfg = wandb_cfg(args)
    run = wandb.init(project=args.output_dir, config=cfg, reinit=True)    
    assert run is wandb.run

    return run


def get_config(filename):
    """ Parse training paremeters from config YAML file. """

    # parse the config file
    with open(filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    config = YamlNamespace(config)

    return config

