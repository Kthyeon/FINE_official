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
from parse_config import ConfigParser
from trainer import DefaultTrainer, TruncatedTrainer, NPCLTrainer
from collections import OrderedDict
import random
import numpy as np



def log_params(conf: OrderedDict, parent_key: str = None):
    for key, value in conf.items():
        if parent_key is not None:
            combined_key = f'{parent_key}-{key}'
        else:
            combined_key = key

        if not isinstance(value, OrderedDict):
            mlflow.log_param(combined_key, value)
        else:
            log_params(value, combined_key)


def main(parse, config: ConfigParser):
    
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
        shuffle=config['data_loader']['args']['shuffle'],
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
    if parse.distillation:
        teacher = config.initialize('arch', module_arch)
        teacher.load_state_dict(torch.load('./asym_40_gce.pth', map_location = 'cpu')['state_dict'])
        for params in teacher.parameters():
            params.requires_grad = False
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
    elif config['train_loss']['type'] == 'NPCLoss':
        train_loss = getattr(module_loss, config['train_loss']['type'])(epsilon=config['train_loss']['args']['epsilon'])
        
    val_loss = getattr(module_loss, config['val_loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = config.initialize('optimizer', torch.optim, [{'params': trainable_params}])

    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    if config['train_loss']['type'] == 'ELRLoss':
        trainer = DefaultTrainer(model, train_loss, metrics, optimizer,
                                     config=config,
                                     data_loader=data_loader,
                                     teacher=teacher,
                                     valid_data_loader=valid_data_loader,
                                     test_data_loader=test_data_loader,
                                     lr_scheduler=lr_scheduler,
                                     val_criterion=val_loss)
    elif config['train_loss']['type'] == 'SCELoss':
        trainer = DefaultTrainer(model, train_loss, metrics, optimizer,
                                     config=config,
                                     data_loader=data_loader,
                                     teacher=teacher,
                                     valid_data_loader=valid_data_loader,
                                     test_data_loader=test_data_loader,
                                     lr_scheduler=lr_scheduler,
                                     val_criterion=val_loss)
    elif config['train_loss']['type'] == 'GCELoss':
        if config['train_loss']['args']['truncated'] == False:
            trainer = DefaultTrainer(model, train_loss, metrics, optimizer,
                                     config=config,
                                     data_loader=data_loader,
                                     teacher=teacher,
                                     valid_data_loader=valid_data_loader,
                                     test_data_loader=test_data_loader,
                                     lr_scheduler=lr_scheduler,
                                     val_criterion=val_loss,
                                     mode = parse.mode)
        elif config['train_loss']['args']['truncated'] == True:
            trainer= TruncatedTrainer(model, train_loss, metrics, optimizer,
                                      config=config,
                                      data_loader=data_loader,
                                     teacher=teacher,
                                      valid_data_loader=valid_data_loader,
                                      test_data_loader=test_data_loader,
                                      lr_scheduler=lr_scheduler,
                                      val_criterion=val_loss)
    elif config['train_loss']['type'] == 'NPCLoss':
        trainer = NPCLTrainer(model, train_loss, metrics, optimizer,
                                     config=config,
                                     data_loader=data_loader,
                                     teacher=teacher,
                                     valid_data_loader=valid_data_loader,
                                     test_data_loader=test_data_loader,
                                     lr_scheduler=lr_scheduler,
                                     val_criterion=val_loss)

    trainer.train()
    logger = config.get_logger('trainer', config['trainer']['verbosity'])
    cfg_trainer = config['trainer']


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--distillation', help='whether to distill knowledge', action='store_true')
    args.add_argument('--mode', type=str, default='ce', choices=['ce', 'kd', 'same'], help = 'distill_type')


    
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
        CustomArgs(['--percent', '--percent'], type=float, target=('trainer', 'percent')),
        CustomArgs(['--asym', '--asym'], type=bool, target=('trainer', 'asym')),
        CustomArgs(['--name', '--exp_name'], type=str, target=('name',)),
        CustomArgs(['--seed', '--seed'], type=int, target=('seed',))
    ]
    
    config = ConfigParser.get_instance(args, options)
    if config['train_loss']['type'] == 'ELRLoss':
        options.append(CustomArgs(['--lamb', '--lamb'], type=float, target=('train_loss', 'args', 'lambda')))
        options.append(CustomArgs(['--beta', '--beta'], type=float, target=('train_loss', 'args', 'beta')))
    elif config['train_loss']['type'] == 'SCELoss':
        options.append(CustomArgs(['--alpha', '--alpha'], type=float, target=('train_loss', 'args', 'alpha')))
        options.append(CustomArgs(['--beta', '--beta'], type=float, target=('train_loss', 'args', 'beta')))
    elif config['train_loss']['type'] == 'GCELoss':
        options.append(CustomArgs(['--q', '--q'], type=float, target=('train_loss', 'args', 'q')))
        options.append(CustomArgs(['--k', '--k'], type=float, target=('train_loss', 'args', 'k')))
        options.append(CustomArgs(['--truncated', '--truncated'], type=bool, target=('train_loss', 'args', 'truncated')))
#     elif config['train_loss']['type'] == ...:
#         options.append(somethings...)
    parse = args.parse_args()
    config = ConfigParser.get_instance(args, options)


    
    ### TRAINING ###
    main(parse, config)
