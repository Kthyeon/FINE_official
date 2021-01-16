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
from trainer import DefaultTrainer, TruncatedTrainer, NPCLTrainer, CoteachingTrainer
from collections import OrderedDict
from trainer.svd_classifier import singular_label, get_out_list, get_singular_value_vector

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
    
    
    wandb.init(config=config, project=parse.project, name=parse.run_name)
    
#     wandb_name = '_sym_' + str(config['trainer']['percent'])
#     wandb_name = '_baseline_' + parse.loss_fn + wandb_name
#     wandb_name = parse.dataset + '_' + parse.lr_scheduler + wandb_name
    
#     wandb.init(config=config, project='noisylabel', entity='goguryeo', name=wandb_name)
    
    
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
    
    wandb.watch(model)
    
    if parse.distillation:
        teacher = config.initialize('arch', module_arch)
        teacher.load_state_dict(torch.load('./checkpoint/' + parse.load_name + '.pth')['state_dict'])
        if not parse.reinit:
            model.load_state_dict(torch.load('./checkpoint/' + parse.load_name + '.pth')['state_dict'])
        
        for params in teacher.parameters():
            params.requires_grad = False
                    
        tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
        singular_dict, v_ortho_dict = get_singular_value_vector(tea_label_list, tea_out_list)

        for key in v_ortho_dict.keys():
            v_ortho_dict[key] = v_ortho_dict[key].cuda()

        teacher_idx = singular_label(v_ortho_dict, tea_out_list, tea_label_list)
        
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
                                     val_criterion=val_loss,
                                     mode = parse.mode,
                                     entropy = parse.entropy,
                                     threshold = parse.threshold
                                )
    elif config['train_loss']['type'] == 'SCELoss':
        trainer = DefaultTrainer(model, train_loss, metrics, optimizer,
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
                                     mode = parse.mode,
                                     entropy = parse.entropy,
                                     threshold = parse.threshold)
        elif config['train_loss']['args']['truncated'] == True:
            trainer= TruncatedTrainer(model, train_loss, metrics, optimizer,
                                      config=config,
                                      data_loader=data_loader,
                                      teacher=teacher,
                                      valid_data_loader=valid_data_loader,
                                      test_data_loader=test_data_loader,
                                      lr_scheduler=lr_scheduler,
                                      val_criterion=val_loss,
                                      mode = parse.mode,
                                      entropy = parse.entropy,
                                      threshold = parse.threshold)
    # coteaching
    elif config['train_loss']['type'] == 'CoteachingLoss':
        
        model1, model2 = config.initialize('arch', module_arch), config.initialize('arch', module_arch)
        
        trainable_params1 = filter(lambda p: p.requires_grad, model1.parameters())
        trainable_params2 = filter(lambda p: p.requires_grad, model2.parameters())

        optimizer1 = config.initialize('optimizer', torch.optim, [{'params': trainable_params1}])
        optimizer2 = config.initialize('optimizer', torch.optim, [{'params': trainable_params2}])
        
        trainer = CoteachingTrainer([model1, model2], train_loss, metrics, [optimizer1, optimizer2],
                                    config=config,
                                    data_loader=data_loader,
                                    teacher=teacher,
                                    valid_data_loader=valid_data_loader,
                                    test_data_loader=test_data_loader,
                                    lr_scheduler=None,
                                    val_criterion=val_loss,
                                    mode=parse.mode,
                                    entropy=parse.entropy,
                                    threshold=parse.threshold,
                                    epoch_decay_start=config['trainer']['epoch_decay_start'],
                                    n_epoch=config['trainer']['epochs'],
                                    learning_rate=config['optimizer']['args']['lr'])
        
    elif config['train_loss']['type'] == 'CoteachingPlusLoss':
        
        model1, model2 = config.initialize('arch', module_arch), config.initialize('arch', module_arch)
        
        trainable_params1 = filter(lambda p: p.requires_grad, model1.parameters())
        trainable_params2 = filter(lambda p: p.requires_grad, model2.parameters())

        optimizer1 = config.initialize('optimizer', torch.optim, [{'params': trainable_params1}])
        optimizer2 = config.initialize('optimizer', torch.optim, [{'params': trainable_params2}])
        
        trainer = CoteachingTrainer([model1, model2], train_loss, metrics, [optimizer1, optimizer2],
                                    config=config,
                                    data_loader=data_loader,
                                    teacher=teacher,
                                    valid_data_loader=valid_data_loader,
                                    test_data_loader=test_data_loader,
                                    lr_scheduler=None,
                                    val_criterion=val_loss,
                                    mode=parse.mode,
                                    entropy=parse.entropy,
                                    threshold=parse.threshold,
                                    epoch_decay_start=config['trainer']['epoch_decay_start'],
                                    n_epoch=config['trainer']['epochs'],
                                    learning_rate=config['optimizer']['args']['lr'])

    trainer.train()
    
    logger = config.get_logger('trainer', config['trainer']['verbosity'])
    cfg_trainer = config['trainer']


if __name__ == '__main__':
#     args = argparse.ArgumentParser(description='PyTorch Template')
#     args.add_argument('-c', '--config', default=None, type=str,
#                       help='config file path (default: None)')
#     args.add_argument('-r', '--resume', default=None, type=str,
#                       help='path to latest checkpoint (default: None)')
#     args.add_argument('-d', '--device', default='1', type=str,
#                       help='indices of GPUs to enable (default: all)')
#     args.add_argument
#     args.add_argument('--distillation', help='whether to distill knowledge', action='store_true')
#     args.add_argument('--mode', type=str, default='ce', choices=['ce', 'same'], help = 'distill_type. same means the same loss of teacher recipe')
#     args.add_argument('--entropy', help='whether to use entropy loss', action='store_true')
#     args.add_argument('--threshold', type=float, default=0.1, help='threshold for the use of entropy loss.')
#     args.add_argument('--wd', type=float, default=5e-4, help = 'weight_decay')
#     args.add_argument('--load_name', type=str, default=None, help = 'teacher checkpoint for distillation')
#     args.add_argument('--reinit', help='whether to use teacher checkpoint', action='store_true')
#     args.add_argument('--project', type=str, default='noisylabel', help='WandB project name')
    
#     args.add_argument('--dataset', type=str, default=None, help='WandB dataset name')
#     args.add_argument('--lr_scheduler', type=str, default=None, help='WandB lr_scheduler name')
#     args.add_argument('--loss_fn', type=str, default=None, help='WandB loss_fn name')
#     args.add_argument('--percent', type=float, default=None, help='Wandb percent')
    
#     parse = args.parse_args()
#     cfg_fname=None
#     if parse.dataset and parse.lr_scheduler and parse.loss_fn:
#         cfg_fname = './hyperparams/' + parse.lr_scheduler + '/config_' + parse.dataset + '_' + parse.loss_fn + '.json'

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--distillation', help='whether to distill knowledge', action='store_true')
    args.add_argument('--mode', type=str, default='ce', choices=['ce', 'same'], help = 'distill_type. same means the same loss of teacher recipe')
    args.add_argument('--entropy', help='whether to use entropy loss', action='store_true')
    args.add_argument('--threshold', type=float, default=0.1, help='threshold for the use of entropy loss.')
    args.add_argument('--wd', type=float, default=5e-4, help = 'weight_decay')
    args.add_argument('--load_name', type=str, default=None, help = 'teacher checkpoint for distillation')
    args.add_argument('--reinit', help='whether to use teacher checkpoint', action='store_true')
    args.add_argument('--project', type=str, default='noisylabel', help='WandB project name')
    args.add_argument('--run_name', type=str, default=None, help='WandB name')
    
    # custom cli options to modify configuration from default values given in json file.
#     CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
#     options = [
#         CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
#         CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
#         CustomArgs(['--name', '--exp_name'], type=str, target=('name',)),
#         CustomArgs(['--seed', '--seed'], type=int, target=('seed',))
#     ]
    
#     config = ConfigParser.get_instance(args, options)

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
        CustomArgs(['--percent', '--percent'], type=float, target=('trainer', 'percent')),
        CustomArgs(['--asym', '--asym'], type=str2bool, target=('trainer', 'asym')),
        CustomArgs(['--name', '--exp_name'], type=str, target=('name',)),
        CustomArgs(['--seed', '--seed'], type=int, target=('seed',))
    ]
    
    
    print (args)
    
    config = ConfigParser.get_instance(args, options)

#     config = ConfigParser.get_instance(args, options, cfg_name=cfg_fname)
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
    elif config['train_loss']['type'] == 'CoteachingLoss':
        options.append(CustomArgs(['--num_gradual', '--num_gradual'], type=float, target=('train_loss', 'args', 'num_gradual')))
    elif config['train_loss']['type'] == 'CoteachingPlusLoss':
        options.append(CustomArgs(['--num_gradual', '--num_gradual'], type=float, target=('train_loss', 'args', 'num_gradual')))

#     elif config['train_loss']['type'] == ...:
#         options.append(somethings...)


    parse = args.parse_args()
    config = ConfigParser.get_instance(args, options)
    
    if parse.percent is not None:
        config['trainer']['percent'] = parse.percent
#     config['trainer']['asym'] = False

    ### TRAINING ###
    main(parse, config)
