import os
import yaml
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from utils.model import CBOW_Model, SkipGram_Model

def get_model_class(model_name:str):
    if model_name == 'cbow':
        return CBOW_Model
    elif model_name == 'skipgram':
        return SkipGram_Model
    else:
        raise ValueError("Choose a model name from: cbow, skipgram")
        return

def get_optimizer_class(name:str):
    if name == 'Adam':
        return optim.Adam
    else:
        raise ValueError("Choose optimizer from: Adam")
        return

def get_lr_scheduler(optimizer, total_epochs:int, verbose:bool = True):
    # Defines and returns a learning rate scheduler.
    # The learning rate decreases linearly from the initial lr set in the optimizer to 0.
    lr_lambda = lambda epoch: (total_epochs - epoch) / total_epochs
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=verbose)
    return lr_scheduler

def save_config(config:dict, model_dir:str):
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

def save_vocab(vocab, model_dir:str):
    vocab_path = os.path.join(model_dir, 'vocab.pt')
    with open(vocab_path, "wb") as f:
        torch.save(vocab, f)