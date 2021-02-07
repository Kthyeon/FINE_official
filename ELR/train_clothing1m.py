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
import model.ImageNet_ResNet_Zoo as module_arch
# import torchvision.models as module_arch
# import model.model as module_arch
from parse_config import ConfigParser
from trainer import DefaultTrainer, TruncatedTrainer, NPCLTrainer, GroundTruthTrainer
from collections import OrderedDict
from trainer.svd_classifier import iterative_eigen, get_out_list, get_singular_value_vector, get_loss_list, isNoisy_ratio

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
#     model = config.initialize('arch', module_arch)
    model = getattr(module_arch, 'resnet50')(pretrained=True,
                                             num_classes=config["num_classes"])
    
    if parse.no_wandb:
        wandb.watch(model)
    
    if parse.distillation:
        teacher = config.initialize('arch', module_arch)
        teacher.load_state_dict(torch.load('./checkpoint/' + parse.load_name)['state_dict'])
        teacher = teacher.cuda()
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
            teacher_idx = get_loss_list(teacher, data_loader)
        print('||||||original||||||')
        isNoisy_ratio(data_loader)
        if parse.second_load_name !=None:
            teacher.load_state_dict(torch.load('./checkpoint/' + parse.second_load_name)['state_dict'])
            teacher = teacher.cuda()
            if not parse.reinit:
                model.load_state_dict(torch.load('./checkpoint/' + parse.second_load_name)['state_dict'])
            for params in teacher.parameters():
                params.requires_grad = False
            if parse.distill_mode == 'eigen':
                tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
                teacher_idx2 = iterative_eigen(1,tea_label_list,tea_out_list,teacher_idx)
            elif parse.distill_mode == 'fulleigen':
                tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
                teacher_idx2 = iterative_eigen(100,tea_label_list,tea_out_list)
            else:
                teacher_idx2 = get_loss_list(teacher, data_loader)
#             print(len(teacher_idx2))
#             teacher_idx = teacher_idx2
            teacher_idx = list(set(teacher_idx) & set(teacher_idx2))
            print('second_distillation')
            if parse.third_load_name !=None:
                teacher.load_state_dict(torch.load('./checkpoint/' + parse.third_load_name)['state_dict'])
                teacher = teacher.cuda()
                if not parse.reinit:
                    model.load_state_dict(torch.load('./checkpoint/' + parse.third_load_name)['state_dict'])
                for params in teacher.parameters():
                    params.requires_grad = False
                if parse.distill_mode == 'eigen':
                    tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
                    teacher_idx3 = iterative_eigen(1,tea_label_list,tea_out_list, teacher_idx)
                elif parse.distill_mode == 'fulleigen':
                    tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
                    teacher_idx3 = iterative_eigen(100,tea_label_list,tea_out_list)
                else:
                    teacher_idx3 = get_loss_list(teacher, data_loader)
                teacher_idx = list(set(teacher_idx) & set(teacher_idx3))
                print('third_ distillation')


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
        print('||||||truncated||||||')
        isNoisy_ratio(data_loader)
        
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
        if config['train_loss']['args']['truncated'] == False:
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
        elif config['train_loss']['args']['truncated'] == True:
            trainer= TruncatedTrainer(model, train_loss, metrics, optimizer,
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
    elif config['train_loss']['type'] == 'GTLoss':
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


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='1', type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--distillation', help='whether to distill knowledge', action='store_true')
    args.add_argument('--distill_mode', type=str, default='eigen', choices=['kmeans','eigen','fulleigen'], help='mode for distillation kmeans or eigen.')
    args.add_argument('--mode', type=str, default='ce', choices=['ce', 'same'], help = 'distill_type. same means the same loss of teacher recipe')
    args.add_argument('--entropy', help='whether to use entropy loss', action='store_true')
    args.add_argument('--threshold', type=float, default=0.1, help='threshold for the use of entropy loss.')
    args.add_argument('--wd', type=float, default=None, help = 'weight_decay')
    args.add_argument('--load_name', type=str, default=None, help = 'teacher checkpoint for distillation')
    args.add_argument('--second_load_name', type=str, default=None, help = '2nd teacher checkpoint for distillation')
    args.add_argument('--third_load_name', type=str, default=None, help = '3rd teacher checkpoint for distillation')
    args.add_argument('--reinit', help='if false, reuse teacher checkpoint', action='store_true')
    
    args.add_argument('--no_wandb', action='store_false', help='if false, not to use wandb')
    # dataset, lr_scheduler, loss_fn are only used to decide config file; they have no effect when config file is given
    args.add_argument('--dataset', type=str, default=None, help='dataset name') 
    args.add_argument('--lr_scheduler', type=str, default=None, help='type of lr_scheduler name')
    args.add_argument('--loss_fn', type=str, default=None, help='loss_fn type name')
    args.add_argument('--arch', type=str, default=None, help='type of model name')
    args.add_argument('--dataseed', type=int, default=123, help='seed for save name')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
        CustomArgs(['--name', '--exp_name'], type=str, target=('name',)),
        CustomArgs(['--seed', '--seed'], type=int, target=('seed',)),
        CustomArgs(['--percent', '--percent'], type=float, target=('trainer', 'percent')),
        CustomArgs(['--asym', '--asym'], type=str2bool, target=('trainer', 'asym')),
    ]
    config = ConfigParser.get_instance(args, options)
    parse = args.parse_args()
    
    ### TRAINING ###
    main(parse, config)