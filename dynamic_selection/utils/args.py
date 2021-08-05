import argparse
import collections
import os
from .parse_config import ConfigParser
from collections import OrderedDict

def parse_args():

    args = argparse.ArgumentParser(description='PyTorch Template')
    
    args.add_argument('-c', '--config', 
                      default=None, 
                      type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', 
                      '--resume', 
                      default=None, 
                      type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', 
                      '--device', 
                      default='1', 
                      type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--distillation', 
                      help='whether to distill knowledge', 
                      action='store_true')
    args.add_argument('--distill_mode', 
                      type=str, 
                      default='eigen', 
                      choices=['fine-kmeans','fine-gmm', 'fine-bmm', 'loss'], 
                      help='mode for distillation kmeans or eigen.')
    args.add_argument('--mode', 
                      type=str, 
                      default='ce', 
                      choices=['ce', 'keeploss'], 
                      help = 'distill_type. keeploss means the same loss of teacher recipe')
    args.add_argument('--entropy', 
                      help='whether to use entropy loss', 
                      action='store_true')
    args.add_argument('--threshold', 
                      type=float, 
                      default=0.1, 
                      help='threshold for the use of entropy loss.')
    args.add_argument('--wd', 
                      type=float, 
                      default=None, 
                      help = 'weight_decay')
    args.add_argument('--load_name', 
                      type=str, 
                      default=None, 
                      help = 'teacher checkpoint for distillation')
#     args.add_argument('--second_load_name', 
#                       type=str, 
#                       default=None, 
#                       help = '2nd teacher checkpoint for distillation')
#     args.add_argument('--third_load_name', 
#                       type=str, 
#                       default=None, 
#                       help = '3rd teacher checkpoint for distillation')
    args.add_argument('--reinit', 
                      help='if false, reuse teacher checkpoint', 
                      action='store_true')
    args.add_argument('--dynamic', 
                      help='if true, dynamic training', 
                      action='store_true')
    
    args.add_argument('--no_wandb', 
                      action='store_false', 
                      help='if false, not to use wandb')
    args.add_argument('--TFT', 
                      action='store_true', 
                      help='True if TFT')
    
    
    # dataset, lr_scheduler, loss_fn are only used to decide config file; they have no effect when config file is given
    args.add_argument('--dataset', 
                      type=str, 
                      default=None, 
                      help='dataset name') 
    args.add_argument('--lr_scheduler', 
                      type=str, 
                      default=None, 
                      help='type of lr_scheduler name')
    args.add_argument('--loss_fn', 
                      type=str, 
                      default=None, 
                      help='loss_fn type name')
    args.add_argument('--arch', 
                      type=str, 
                      default=None, 
                      help='type of model name')
    args.add_argument('--dataseed', 
                      type=int, 
                      default=123, 
                      help='seed for save name')
    args.add_argument('--traintools', 
                      type=str, 
                      default='robustloss', 
                      choices=['robustloss', 'robustlossgt', 'coteaching', 'trainingclothing1m'])
    
    # Arguments for Dynamic Training
    args.add_argument('--warmup',
                      type=int,
                      default=40,
                      help='warm-up epochs')
    
    args.add_argument('--every',
                      type=int,
                      default=10,
                      help='proceed FINE frequent')
    args.add_argument('--zeta',
                      type=float,
                      default=0.5,
                      help='hyperparameter for fine')


    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
        CustomArgs(['--name', '--exp_name'], type=str, target=('name',)),
        CustomArgs(['--seed', '--seed'], type=int, target=('seed',)),
        CustomArgs(['--percent', '--percent'], type=float, target=('trainer', 'percent')),
        CustomArgs(['--asym', '--asym'], type=str2bool, target=('trainer', 'asym')),
        CustomArgs(['--instance', '--instance'], type=str2bool, target=('trainer', 'instance'))
    ]
    
    os.chdir(os.path.join(os.getcwd(), 'dynamic_selection'))
    
    config = ConfigParser.get_instance(args, options)
    parse = args.parse_args()
    
    return config, parse

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