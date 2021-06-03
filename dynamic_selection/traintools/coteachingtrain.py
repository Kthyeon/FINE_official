import argparse
import collections
import sys
import requests
import socket
import torch
import mlflow
import mlflow.pytorch
import data_loader.data_loaders as module_data
import loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from trainer import CoteachingTrainer, FCoteachingTrainer
from collections import OrderedDict

from selection.svd_classifier import *
from selection.gmm import *
from selection.util import *

from utils.parse_config import ConfigParser
from utils.util import *
from utils.args import *

import random
import numpy as np
import copy

import wandb

__all__ = ['coteachingtrain']

def coteachingtrain(parse, config: ConfigParser):
    # implementation for WandB
    wandb_run_name_list = wandbRunlist(config, parse)
    
    if parse.no_wandb:
        wandb.init(config=config, project='noisylabel', entity='goguryeo', name=wandb_run_name)
    
    # By default, pytorch utilizes multi-threaded cpu
    numthread = torch.get_num_threads()
    torch.set_num_threads(numthread)    
    logger = config.get_logger('train')
    
    # Set seed for reproducibility
    fix_seed(config['seed'])
    
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size= config['data_loader']['args']['batch_size'],
        shuffle=False if parse.distillation else config['data_loader']['args']['shuffle'],
        validation_split=0.0,
        num_batches=config['data_loader']['args']['num_batches'],
        training=True,
        num_workers=config['data_loader']['args']['num_workers'],
        pin_memory=config['data_loader']['args']['pin_memory'],
        seed=parse.dataseed # parse.seed
    )


    valid_data_loader = None
    test_data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=128,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    ).split_validation()

    # build model architecture, then print to console
    model = config.initialize('arch', module_arch)
    
    if parse.no_wandb: wandb.watch(model)
    
    if parse.distillation:
        teacher = config.initialize('teacher_arch', module_arch)

        data_loader = getattr(module_data, config['data_loader']['type'])(
            config['data_loader']['args']['data_dir'],
            batch_size= config['data_loader']['args']['batch_size'],
            shuffle=config['data_loader']['args']['shuffle'],
            validation_split=0.0,
            num_batches=config['data_loader']['args']['num_batches'],
            training=True,
            num_workers=config['data_loader']['args']['num_workers'],
            pin_memory=config['data_loader']['args']['pin_memory'],
            seed=parse.dataseed,
            teacher_idx = extract_cleanidx(teacher, data_loader, parse))
    else:
        teacher = None

    # get function handles of loss and metrics
    logger.info(config.config)
    if hasattr(data_loader.dataset, 'num_raw_example'):
        num_examp = data_loader.dataset.num_raw_example
    else:
        num_examp = len(data_loader.dataset)
    
    # F-coteaching
    if config['train_loss']['type'] == 'CCELoss':
        train_loss = getattr(module_loss, 'CCELoss')()
        
    # coteaching
    elif config['train_loss']['type'] == 'CoteachingLoss':
        train_loss = getattr(module_loss, 'CoteachingLoss')(forget_rate=config['trainer']['percent'],
                                                            num_gradual=int(config['train_loss']['args']['num_gradual']),
                                                            n_epoch=config['trainer']['epochs'])

    # coteaching_plus
    elif config['train_loss']['type'] == 'CoteachingPlusLoss':
        train_loss = getattr(module_loss, 'CoteachingPlusLoss')(forget_rate=config['trainer']['percent'],
                                                                num_gradual=int(config['train_loss']['args']['num_gradual']),
                                                                n_epoch=config['trainer']['epochs'])

        
    val_loss = getattr(module_loss, config['val_loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    
    # F-coteaching
    if config['train_loss']['type'] == 'CCELoss':
        
        model = config.initialize('arch', module_arch)
        trainer = FCoteachingTrainer(model, train_loss, metrics, None,
                                       config=config,
                                       data_loader=data_loader,
                                       parse=parse,
                                       teacher=teacher,
                                       valid_data_loader=valid_data_loader,
                                       test_data_loader=test_data_loader,
                                       lr_scheduler=None,
                                       val_criterion=val_loss,
                                       mode = parse.mode,
                                       entropy = parse.entropy,
                                       threshold = parse.threshold
                                  )
    
    # coteaching
    elif config['train_loss']['type'] == 'CoteachingLoss':
        
        model1, model2 = config.initialize('arch', module_arch), config.initialize('arch', module_arch)
        
        trainable_params1 = filter(lambda p: p.requires_grad, model1.parameters())
        trainable_params2 = filter(lambda p: p.requires_grad, model2.parameters())

        optimizer1 = config.initialize('optimizer', torch.optim, [{'params': trainable_params1}])
        optimizer2 = config.initialize('optimizer', torch.optim, [{'params': trainable_params2}])
        
        if isinstance(optimizer1, torch.optim.Adam):
            lr_scheduler = None
        else:
            lr_scheduler1 = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer1)
            lr_scheduler2 = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer2)
            lr_scheduler = [lr_scheduler1, lr_scheduler2]
        
#         print ('$$$$$$$$$$$$$$$')
#         print (config['optimizer'])
        
        trainer = CoteachingTrainer([model1, model2], train_loss, metrics, [optimizer1, optimizer2],
                                    config=config,
                                    data_loader=data_loader,
                                    parse=parse,
                                    teacher=teacher,
                                    valid_data_loader=valid_data_loader,
                                    test_data_loader=test_data_loader,
                                    lr_scheduler=lr_scheduler,
                                    val_criterion=val_loss,
                                    mode=parse.mode,
                                    entropy=parse.entropy,
                                    threshold=parse.threshold,
                                    epoch_decay_start=config['trainer']['epoch_decay_start'],
                                    n_epoch=config['trainer']['epochs'],
                                    learning_rate=config['optimizer']['args']['lr']
                                   )
        
    elif config['train_loss']['type'] == 'CoteachingPlusLoss':
        
        model1, model2 = config.initialize('arch', module_arch), config.initialize('arch', module_arch)
        
        trainable_params1 = filter(lambda p: p.requires_grad, model1.parameters())
        trainable_params2 = filter(lambda p: p.requires_grad, model2.parameters())

        optimizer1 = config.initialize('optimizer', torch.optim, [{'params': trainable_params1}])
        optimizer2 = config.initialize('optimizer', torch.optim, [{'params': trainable_params2}])
        
        if isinstance(optimizer1, torch.optim.Adam):
            lr_scheduler = None
        else:
            lr_scheduler1 = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer1)
            lr_scheduler2 = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer2)
            lr_scheduler = [lr_scheduler1, lr_scheduler2]
        
        trainer = CoteachingTrainer([model1, model2], train_loss, metrics, [optimizer1, optimizer2],
                                    config=config,
                                    data_loader=data_loader,
                                    parse=parse,
                                    teacher=teacher,
                                    valid_data_loader=valid_data_loader,
                                    test_data_loader=test_data_loader,
                                    lr_scheduler=lr_scheduler,
                                    val_criterion=val_loss,
                                    mode=parse.mode,
                                    entropy=parse.entropy,
                                    threshold=parse.threshold,
                                    epoch_decay_start=config['trainer']['epoch_decay_start'],
                                    n_epoch=config['trainer']['epochs'],
                                    learning_rate=config['optimizer']['args']['lr']
                                   )
        
    elif config['train_loss']['type'] == 'CoteachingDistillLoss':
        
        model1, model2 = config.initialize('arch', module_arch), config.initialize('arch', module_arch)
        
        trainable_params1 = filter(lambda p: p.requires_grad, model1.parameters())
        trainable_params2 = filter(lambda p: p.requires_grad, model2.parameters())

        optimizer1 = config.initialize('optimizer', torch.optim, [{'params': trainable_params1}])
        optimizer2 = config.initialize('optimizer', torch.optim, [{'params': trainable_params2}])
        
        if isinstance(optimizer1, torch.optim.Adam):
            lr_scheduler = None
        else:
            lr_scheduler1 = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer1)
            lr_scheduler2 = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer2)
            lr_scheduler = [lr_scheduler1, lr_scheduler2]
            
        trainer = CoteachingTrainer([model1, model2], train_loss, metrics, [optimizer1, optimizer2],
                                    config=config,
                                    data_loader=data_loader,
                                    parse=parse,
                                    teacher=teacher,
                                    valid_data_loader=valid_data_loader,
                                    test_data_loader=test_data_loader,
                                    lr_scheduler=lr_scheduler,
                                    val_criterion=val_loss,
                                    mode=parse.mode,
                                    entropy=parse.entropy,
                                    threshold=parse.threshold,
                                    epoch_decay_start=config['trainer']['epoch_decay_start'],
                                    n_epoch=config['trainer']['epochs'],
                                    learning_rate=config['optimizer']['args']['lr']
                                   )
        
    elif config['train_loss']['type'] == 'CoteachingPlusDistillLoss':
        
        model1, model2 = config.initialize('arch', module_arch), config.initialize('arch', module_arch)
        
        trainable_params1 = filter(lambda p: p.requires_grad, model1.parameters())
        trainable_params2 = filter(lambda p: p.requires_grad, model2.parameters())

        optimizer1 = config.initialize('optimizer', torch.optim, [{'params': trainable_params1}])
        optimizer2 = config.initialize('optimizer', torch.optim, [{'params': trainable_params2}])
        
        if isinstance(optimizer1, torch.optim.Adam):
            lr_scheduler = None
        else:
            lr_scheduler1 = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer1)
            lr_scheduler2 = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer2)
            lr_scheduler = [lr_scheduler1, lr_scheduler2]
            
        trainer = CoteachingTrainer([model1, model2], train_loss, metrics, [optimizer1, optimizer2],
                                    config=config,
                                    data_loader=data_loader,
                                    parse=parse,
                                    teacher=teacher,
                                    valid_data_loader=valid_data_loader,
                                    test_data_loader=test_data_loader,
                                    lr_scheduler=lr_scheduler,
                                    val_criterion=val_loss,
                                    mode=parse.mode,
                                    entropy=parse.entropy,
                                    threshold=parse.threshold,
                                    epoch_decay_start=config['trainer']['epoch_decay_start'],
                                    n_epoch=config['trainer']['epochs'],
                                    learning_rate=config['optimizer']['args']['lr']
                                   )
        
    trainer.train()
    
    logger = config.get_logger('trainer', config['trainer']['verbosity'])
    cfg_trainer = config['trainer']