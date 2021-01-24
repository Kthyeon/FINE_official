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
from trainer import CoteachingTrainer
from collections import OrderedDict
from trainer.svd_classifier import iterative_eigen, get_out_list, get_singular_value_vector, get_loss_list, get_loss_list_2d, isNoisy_ratio

import random
import numpy as np
import copy

import wandb

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
        teacher = config.initialize('teacher_arch', module_arch)
        teacher.load_state_dict(torch.load('./checkpoint/' + parse.load_name)['state_dict'])
        if not parse.reinit:
            model.load_state_dict(torch.load('./checkpoint/' + parse.load_name)['state_dict'])
        for params in teacher.parameters():
            params.requires_grad = False
        if parse.distill_mode == 'eigen':
            tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
            teacher_idx = iterative_eigen(1,tea_label_list,tea_out_list)
        elif parse.distill_mode == 'fulleigen':
            tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
            teacher_idx = iterative_eigen(100,tea_label_list,tea_out_list)
        else:
            teacher_idx = get_loss_list_2d(teacher, data_loader, n_clusters=3)
        
#         print('||||||original||||||')
#         isNoisy_ratio(data_loader)
        
        data_loader = getattr(module_data, config['data_loader']['type'])(
            config['data_loader']['args']['data_dir'],
            batch_size= config['data_loader']['args']['batch_size'],
            shuffle=config['data_loader']['args']['shuffle'],
            validation_split=0.0,
            num_batches=config['data_loader']['args']['num_batches'],
            training=True,
            num_workers=config['data_loader']['args']['num_workers'],
            pin_memory=config['data_loader']['args']['pin_memory'],
#             teacher_idx = teacher_idx
        )
#         print('||||||truncated||||||')
#         isNoisy_ratio(data_loader)
        
    else:
        teacher = None
        
    print ('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    
    print (len(teacher_idx))
    
    num_examp = len(data_loader.train_dataset)
    real = torch.Tensor([True for _ in range(num_examp)])
    pred = torch.Tensor([False for _ in range(num_examp)])
    real[torch.Tensor(data_loader.train_dataset.noise_indx).long()] = False
    pred[teacher_idx] = True
    
    true_positive = torch.Tensor([pred[i].long() & real[i].long() for i in range(len(real))])
    print (torch.sum(true_positive) / torch.sum(pred))
    
    print ('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

    # get function handles of loss and metrics
    logger.info(config.config)
    if hasattr(data_loader.dataset, 'num_raw_example'):
        num_examp = data_loader.dataset.num_raw_example
    else:
        num_examp = len(data_loader.dataset)
    
    # coteaching
    if config['train_loss']['type'] == 'CoteachingLoss':
        train_loss = getattr(module_loss, 'CoteachingLoss')(forget_rate=config['trainer']['percent'],
                                                            num_gradual=int(config['train_loss']['args']['num_gradual']),
                                                            n_epoch=config['trainer']['epochs'])

    # coteaching_plus
    elif config['train_loss']['type'] == 'CoteachingPlusLoss':
        train_loss = getattr(module_loss, 'CoteachingPlusLoss')(forget_rate=config['trainer']['percent'],
                                                                num_gradual=int(config['train_loss']['args']['num_gradual']),
                                                                n_epoch=config['trainer']['epochs'])
    
    # coteaching + winning_ticket!
    elif config['train_loss']['type'] == 'CoteachingDistillLoss':
        train_loss = getattr(module_loss, 'CoteachingDistillLoss')(forget_rate=config['trainer']['percent'],
                                                                   num_gradual=int(config['train_loss']['args']['num_gradual']),
                                                                   n_epoch=config['trainer']['epochs'],
                                                                   num_examp=num_examp,
                                                                   clean_indexs=teacher_idx)
        
    # coteaching_plus + winning_ticket!
    elif config['train_loss']['type'] == 'CoteachingPlusDistillLoss':
        train_loss = getattr(module_loss, 'CoteachingPlusDistillLoss')(forget_rate=config['trainer']['percent'],
                                                                       num_gradual=int(config['train_loss']['args']['num_gradual']),
                                                                       n_epoch=config['trainer']['epochs'],
                                                                       num_examp=num_examp,
                                                                       clean_indexs=teacher_idx)

        
    val_loss = getattr(module_loss, config['val_loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # coteaching
    if config['train_loss']['type'] == 'CoteachingLoss':
        
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
        
        print (config['optimizer'])
        
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


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='1', type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--distillation', help='whether to distill knowledge', action='store_true')
    args.add_argument('--distill_mode', type=str, default='eigen', choices=['kmeans','eigen', 'fulleigen'], help='mode for distillation kmeans or eigen.')
    args.add_argument('--mode', type=str, default='ce', choices=['ce', 'same'], help = 'distill_type. same means the same loss of teacher recipe')
    args.add_argument('--entropy', help='whether to use entropy loss', action='store_true')
    args.add_argument('--threshold', type=float, default=0.1, help='threshold for the use of entropy loss.')
    args.add_argument('--wd', type=float, default=None, help = 'weight_decay')
    args.add_argument('--load_name', type=str, default=None, help = 'teacher checkpoint for distillation')
    args.add_argument('--reinit', help='if false, reuse teacher checkpoint', action='store_true')
    
    args.add_argument('--no_wandb', action='store_false', help='if false, not to use wandb')
    # dataset, lr_scheduler, loss_fn are only used to decide config file; they have no effect when config file is given
    args.add_argument('--dataset', type=str, default=None, help='dataset name') 
    args.add_argument('--lr_scheduler', type=str, default=None, help='type of lr_scheduler name')
    args.add_argument('--loss_fn', type=str, default=None, help='loss_fn type name')
    args.add_argument('--arch', type=str, default=None, help='type of model name')
    args.add_argument('--teacher_arch', type=str, default=None, help='type of teacher model name')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
        CustomArgs(['--name', '--exp_name'], type=str, target=('name',)),
        CustomArgs(['--seed', '--seed'], type=int, target=('seed',)),
        CustomArgs(['--percent', '--percent'], type=float, target=('trainer', 'percent')),
        CustomArgs(['--asym', '--asym'], type=str2bool, target=('trainer', 'asym')),
        CustomArgs(['--num_gradual', '--num_gradual'], type=int, target=('train_loss', 'args', 'num_gradual'))
    ]
    config = ConfigParser.get_instance(args, options)
    parse = args.parse_args()
    
    ### TRAINING ###
    main(parse, config)