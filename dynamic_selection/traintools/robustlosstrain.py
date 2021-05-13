import sys
import requests
import socket
import random
import numpy as np
import copy
import torch

from collections import OrderedDict

import data_loader.data_loaders as module_data
import loss as module_loss

import model.metric as module_metric
import model.model as module_arch

from trainer import DefaultTrainer, TruncatedTrainer, GroundTruthTrainer, DynamicTrainer

from selection.svd_classifier import *
from selection.gmm import *
from selection.util import *

from utils.parse_config import ConfigParser
from utils.util import *
from utils.args import *

import wandb


__all__ = ['robustlosstrain']


def robustlosstrain(parse, config: ConfigParser):
    # implementation for WandB
    wandb_run_name_list = wandbRunlist(config, parse)
    
    if parse.no_wandb:
        wandb.init(config=config, project='noisylabel', entity='goguryeo', name='_'.join(wandb_run_name_list))
    
    # By default, pytorch utilizes multi-threaded cpu
    numthread = torch.get_num_threads()
    torch.set_num_threads(numthread)
    logger = config.get_logger('train')
    
    # Set seed for reproducibility
    fix_seed(config['seed'])
    
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size= config['data_loader']['args']['batch_size'],
        shuffle=False if parse.distillation else config['data_loader']['args']['shuffle'] ,
        validation_split=0.1,
        num_batches=config['data_loader']['args']['num_batches'],
        training=True,
        num_workers=config['data_loader']['args']['num_workers'],
        pin_memory=config['data_loader']['args']['pin_memory'],
        seed=parse.dataseed
    )

    valid_data_loader = data_loader.split_validation()

#     valid_data_loader = None
    
    # test_data_loader = None

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
        teacher = config.initialize('arch', module_arch)
        
        data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size= config['data_loader']['args']['batch_size'],
        shuffle=config['data_loader']['args']['shuffle'],
#         validation_split=config['data_loader']['args']['validation_split'],
        validation_split=0.1,
        num_batches=config['data_loader']['args']['num_batches'],
        training=True,
        num_workers=config['data_loader']['args']['num_workers'],
        pin_memory=config['data_loader']['args']['pin_memory'],
        teacher_idx = extract_cleanidx(teacher, data_loader, parse),
        seed = parse.dataseed)
    else:
        teacher = None

    # get function handles of loss and metrics
    logger.info(config.config)
    if hasattr(data_loader.dataset, 'num_raw_example'):
        num_examp = data_loader.dataset.num_raw_example
    else:
        num_examp = len(data_loader.dataset)
    
    if config['train_loss']['type'] == 'ELRLoss':
        train_loss = getattr(module_loss, 'ELRLoss')(num_examp=num_examp,
                                                     num_classes=config['num_classes'],
                                                     beta=config['train_loss']['args']['beta'])
    elif config['train_loss']['type'] == 'SCELoss':
        train_loss = getattr(module_loss, 'SCELoss')(alpha=config['train_loss']['args']['alpha'],
                                                     beta=config['train_loss']['args']['beta'],
                                                     num_classes=config['num_classes'])
    elif config['train_loss']['type'] == 'GCELoss':
        train_loss = getattr(module_loss, 'GCELoss')(q=config['train_loss']['args']['q'],
                                                     k=config['train_loss']['args']['k'],
                                                     trainset_size=num_examp,
                                                     truncated=config['train_loss']['args']['truncated'])
    elif config['train_loss']['type'] == 'GTLoss':
        train_loss = getattr(module_loss, 'GTLoss')()
        
    else:
        train_loss = getattr(module_loss, 'CCELoss')()

        
    val_loss = getattr(module_loss, config['val_loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = config.initialize('optimizer', torch.optim, [{'params': trainable_params}])

    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    if config['train_loss']['type'] == 'ELRLoss':
        if parse.dynamic:
            trainer = DynamicTrainer(model, train_loss, metrics, optimizer,
                                         config=config,
                                         data_loader=data_loader,
                                         parse=parse,
                                         teacher=teacher,
                                         valid_data_loader=valid_data_loader,
                                         test_data_loader=test_data_loader,
                                         lr_scheduler=lr_scheduler,
                                         val_criterion=val_loss,
                                         mode = parse.mode,
                                         entropy = parse.entropy,
                                         threshold = parse.threshold
                                    )
        else:
            trainer = DefaultTrainer(model, train_loss, metrics, optimizer,
                                         config=config,
                                         data_loader=data_loader,
                                         parse=parse,
                                         teacher=teacher,
                                         valid_data_loader=valid_data_loader,
                                         test_data_loader=test_data_loader,
                                         lr_scheduler=lr_scheduler,
                                         val_criterion=val_loss,
                                         mode = parse.mode,
                                         entropy = parse.entropy,
                                         threshold = parse.threshold
                                    )
    elif config['train_loss']['type'] == 'SCELoss':
        if parse.dynamic:
            trainer = DynamicTrainer(model, train_loss, metrics, optimizer,
                                         config=config,
                                         data_loader=data_loader,
                                         parse=parse,
                                         teacher=teacher,
                                         valid_data_loader=valid_data_loader,
                                         test_data_loader=test_data_loader,
                                         lr_scheduler=lr_scheduler,
                                         val_criterion=val_loss,
                                         mode = parse.mode,
                                         entropy = parse.entropy,
                                         threshold = parse.threshold
                                    )
        else:
            trainer = DefaultTrainer(model, train_loss, metrics, optimizer,
                                         config=config,
                                         data_loader=data_loader,
                                         parse=parse,
                                         teacher=teacher,
                                         valid_data_loader=valid_data_loader,
                                         test_data_loader=test_data_loader,
                                         lr_scheduler=lr_scheduler,
                                         val_criterion=val_loss,
                                         mode = parse.mode,
                                         entropy = parse.entropy,
                                         threshold = parse.threshold
                                    )
            
    elif config['train_loss']['type'] == 'GCELoss':
        if parse.dynamic:
            trainer = DynamicTrainer(model, train_loss, metrics, optimizer,
                                     config=config,
                                     data_loader=data_loader,
                                     parse=parse,
                                     teacher=teacher,
                                     valid_data_loader=valid_data_loader,
                                     test_data_loader=test_data_loader,
                                     lr_scheduler=lr_scheduler,
                                     val_criterion=val_loss,
                                     mode = parse.mode,
                                     entropy = parse.entropy,
                                     threshold = parse.threshold
                                    )
        else:
            trainer = DefaultTrainer(model, train_loss, metrics, optimizer,
                                     config=config,
                                     data_loader=data_loader,
                                     parse=parse,
                                     teacher=teacher,
                                     valid_data_loader=valid_data_loader,
                                     test_data_loader=test_data_loader,
                                     lr_scheduler=lr_scheduler,
                                     val_criterion=val_loss,
                                     mode = parse.mode,
                                     entropy = parse.entropy,
                                     threshold = parse.threshold
                                    )
#     elif config['train_loss']['type'] == 'GTLoss':
#         trainer = GroundTruthTrainer(model, train_loss, metrics, optimizer,
#                                      config=config,
#                                      data_loader=data_loader,
#                                      teacher=teacher,
#                                      valid_data_loader=valid_data_loader,
#                                      test_data_loader=test_data_loader,
#                                      lr_scheduler=lr_scheduler,
#                                      val_criterion=val_loss,
#                                      mode = parse.mode,
#                                      entropy = parse.entropy,
#                                      threshold = parse.threshold
#                                     )
    elif config['train_loss']['type'] == 'CCELoss':
        if parse.dynamic:
            trainer = DynamicTrainer(model, train_loss, metrics, optimizer,
                                       config=config,
                                       data_loader=data_loader,
                                       parse=parse,
                                       teacher=teacher,
                                       valid_data_loader=valid_data_loader,
                                       test_data_loader=test_data_loader,
                                       lr_scheduler=lr_scheduler,
                                       val_criterion=val_loss,
                                       mode = parse.mode,
                                       entropy = parse.entropy,
                                       threshold = parse.threshold
                                  )
        
        else:
            trainer = DefaultTrainer(model, train_loss, metrics, optimizer,
                                       config=config,
                                       data_loader=data_loader,
                                       parse=parse,
                                       teacher=teacher,
                                       valid_data_loader=valid_data_loader,
                                       test_data_loader=test_data_loader,
                                       lr_scheduler=lr_scheduler,
                                       val_criterion=val_loss,
                                       mode = parse.mode,
                                       entropy = parse.entropy,
                                       threshold = parse.threshold
                                  )
    

    trainer.train()
    
    logger = config.get_logger('trainer', config['trainer']['verbosity'])
    cfg_trainer = config['trainer']


# if __name__ == '__main__':
#     config, parse = parse_args()
    
#     ### TRAINING ###
#     main(parse, config)
