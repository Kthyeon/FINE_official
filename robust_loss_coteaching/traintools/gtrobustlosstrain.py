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
from trainer import GroundTruthTrainer
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

# from trainer import DefaultTrainer, TruncatedTrainer, GroundTruthTrainer, DynamicTrainer
# from trainer.svd_classifier import iterative_eigen, get_out_list, get_singular_value_vector, get_loss_list, isNoisy_ratio, kmean_eigen_out, topk_eigen_kmean, extract_teacherIdx
from selection.svd_classifier import *

from utils.util import *
from utils.args import *

__all__=['gtrobustlosstrain']


def gtrobustlosstrain(parse, config: ConfigParser):
    dataset_name = config['name'].split('_')[0]
    lr_scheduler_name = config['lr_scheduler']['type']
    loss_fn_name = config['train_loss']['type']
    
    wandb_run_name_list = []
    
    if parse.distillation:
        if parse.distill_mode == 'eigen':
            wandb_run_name_list.append('distil')
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
    
    if parse.no_wandb:
        wandb.init(config=config, project='noisylabel', entity='goguryeo', name=wandb_run_name)
    
    # By default, pytorch utilizes multi-threaded cpu
    # Set to handle whole procedures on a single core
    torch.set_num_threads(1)
    
    logger = config.get_logger('train')
    
    # Set seed for reproducibility
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True
    np.random.seed(config['seed'])
    
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size= config['data_loader']['args']['batch_size'],
        shuffle=False if parse.distillation else config['data_loader']['args']['shuffle'] ,
#         validation_split=config['data_loader']['args']['validation_split'],
        validation_split=0.0,
        num_batches=config['data_loader']['args']['num_batches'],
        training=True,
        num_workers=config['data_loader']['args']['num_workers'],
        pin_memory=config['data_loader']['args']['pin_memory']
    )

    # valid_data_loader = data_loader.split_validation()

    valid_data_loader = None
    
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
    
    if parse.no_wandb:
        wandb.watch(model)
    
    if parse.distillation:
        teacher = config.initialize('arch', module_arch)
        teacher.load_state_dict(torch.load('./checkpoint/' + parse.load_name)['state_dict'])
        if not parse.reinit:
            model.load_state_dict(torch.load('./checkpoint/' + parse.load_name)['state_dict'])
        for params in teacher.parameters():
                params.requires_grad = False
        if parse.distill_mode == 'eigen':
            tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
            singular_dict, v_ortho_dict = get_singular_value_vector(tea_label_list, tea_out_list)

            for key in v_ortho_dict.keys():
                v_ortho_dict[key] = v_ortho_dict[key].cuda()

            teacher_idx = singular_label(v_ortho_dict, tea_out_list, tea_label_list)
        else:
            teacher_idx = get_out_list(teacher, data_loader)
        
        data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size= config['data_loader']['args']['batch_size'],
        shuffle=config['data_loader']['args']['shuffle'],
#         validation_split=config['data_loader']['args']['validation_split'],
        validation_split=0.0,
        num_batches=config['data_loader']['args']['num_batches'],
        training=True,
        num_workers=config['data_loader']['args']['num_workers'],
        pin_memory=config['data_loader']['args']['pin_memory'],
        teacher_idx = teacher_idx)
    else:
        teacher = None

    # get function handles of loss and metrics
    logger.info(config.config)
    if hasattr(data_loader.dataset, 'num_raw_example'):
        num_examp = data_loader.dataset.num_raw_example
    else:
        num_examp = len(data_loader.dataset)
    
    if config['train_loss']['type'] == 'ELR_GTLoss':
        train_loss = getattr(module_loss, 'ELR_GTLoss')(num_examp=num_examp,
                                                     num_classes=config['num_classes'],
                                                     beta=config['train_loss']['args']['beta'])
    elif config['train_loss']['type'] == 'SCE_GTLoss':
        train_loss = getattr(module_loss, 'SCE_GTLoss')(alpha=config['train_loss']['args']['alpha'],
                                                     beta=config['train_loss']['args']['beta'],
                                                     num_classes=config['num_classes'])
    elif config['train_loss']['type'] == 'GCE_GTLoss':
        train_loss = getattr(module_loss, 'GCE_GTLoss')(q=config['train_loss']['args']['q'],
                                                     k=config['train_loss']['args']['k'],
                                                     trainset_size=num_examp,
                                                     truncated=config['train_loss']['args']['truncated'])
    elif config['train_loss']['type'] == 'CCE_GTLoss':
        train_loss = getattr(module_loss, 'CCE_GTLoss')()

        
    val_loss = getattr(module_loss, config['val_loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = config.initialize('optimizer', torch.optim, [{'params': trainable_params}])

    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    if config['train_loss']['type'] == 'ELR_GTLoss':
        trainer = GroundTruthTrainer(model, train_loss, metrics, optimizer,
                                     config=config,
                                     data_loader=data_loader,
                                     teacher=teacher,
                                     valid_data_loader=valid_data_loader,
                                     test_data_loader=test_data_loader,
                                     lr_scheduler=lr_scheduler,
                                     val_criterion=val_loss,
                                     mode = parse.mode,
                                     entropy = parse.entropy,
                                     threshold = parse.threshold
                                    )
    elif config['train_loss']['type'] == 'SCE_GTLoss':
        trainer = GroundTruthTrainer(model, train_loss, metrics, optimizer,
                                     config=config,
                                     data_loader=data_loader,
                                     teacher=teacher,
                                     valid_data_loader=valid_data_loader,
                                     test_data_loader=test_data_loader,
                                     lr_scheduler=lr_scheduler,
                                     val_criterion=val_loss,
                                     mode = parse.mode,
                                     entropy = parse.entropy,
                                     threshold = parse.threshold
                                    )
    elif config['train_loss']['type'] == 'GCE_GTLoss':
        trainer = GroundTruthTrainer(model, train_loss, metrics, optimizer,
                                     config=config,
                                     data_loader=data_loader,
                                     teacher=teacher,
                                     valid_data_loader=valid_data_loader,
                                     test_data_loader=test_data_loader,
                                     lr_scheduler=lr_scheduler,
                                     val_criterion=val_loss,
                                     mode = parse.mode,
                                     entropy = parse.entropy,
                                     threshold = parse.threshold
                                    )
    elif config['train_loss']['type'] == 'CCE_GTLoss':
        trainer = GroundTruthTrainer(model, train_loss, metrics, optimizer,
                                     config=config,
                                     data_loader=data_loader,
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