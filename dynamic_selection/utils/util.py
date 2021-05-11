import json
from pathlib import Path
from datetime import datetime
from itertools import repeat
from collections import OrderedDict
import numpy as np

import random
import torch

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=True)


def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()



def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from  2"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    current = np.clip(current, 0.0, rampdown_length)
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def cosine_rampup(current, rampup_length):
    """Cosine rampup"""
    current = np.clip(current, 0.0, rampup_length)
    return float(-.5 * (np.cos(np.pi * current / rampup_length) - 1))

def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

def wandbRunlist(config, parse):
    dataset_name = config['name'].split('_')[0]
    lr_scheduler_name = config['lr_scheduler']['type']
    loss_fn_name = config['train_loss']['type']
    
    wandb_run_name_list = []
    
    if parse.distillation:
        if parse.distill_mode == 'eigen':
            wandb_run_name_list.append('distil')
        elif parse.distill_mode == 'fulleigen':
            wandb_run_name_list.append('fulldistill')
        else:
            wandb_run_name_list.append('kmeans')
    else:
        wandb_run_name_list.append('baseline')
    wandb_run_name_list.append(dataset_name)
    wandb_run_name_list.append(lr_scheduler_name)
    wandb_run_name_list.append(loss_fn_name)
    wandb_run_name_list.append(str(config['trainer']['asym']))
    wandb_run_name_list.append(str(config['trainer']['percent']))
    wandb_run_name = '_'.join(wandb_run_name_list)
    
    return wandb_run_name_list

