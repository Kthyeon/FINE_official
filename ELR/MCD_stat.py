import argparse
import torch
import sys
import os
import json
import random
import numpy as np
import collections
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt 
import sklearn.covariance
import scipy
import pdb

import data_loader.data_loaders as module_data
import loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import torch.nn as nn
import torch.nn.functional as F
import model.model as module_arch

from sklearn import metrics
from sklearn import cluster
from tqdm import tqdm
from torch.autograd import Variable
from parse_config import ConfigParser
from collections import OrderedDict
from utils.MCD_utils import *

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

    #Set seed for reproducibility
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True
    np.random.seed(config['seed'])
    saved_path = './saved' 
    tmp_asym = 'asym' if config['trainer']['asym'] else 'sym'
    data_name, net_fam = config['name'].split('_')[0], config['name'].split('_')[1]

    if not os.path.isdir(saved_path):
        os.mkdir(saved_path)
    next_path = os.path.join(saved_path, 'mahalanobis')
    if not os.path.isdir(next_path):
        os.mkdir(next_path)
    next_path = os.path.join(next_path, data_name)
    if not os.path.isdir(next_path):
        os.mkdir(next_path)
    next_path = os.path.join(next_path, net_fam)
    if not os.path.isdir(next_path):
        os.mkdir(next_path)
    next_path = os.path.join(next_path, config['lr_scheduler']['type'])
    if not os.path.isdir(next_path):
        os.mkdir(next_path)
    next_path = os.path.join(next_path, config['train_loss']['type'])
    if not os.path.isdir(next_path):
        os.mkdir(next_path)
    next_path = os.path.join(next_path, tmp_asym)
    if not os.path.isdir(next_path):
        os.mkdir(next_path)
    file_root = os.path.join(next_path, str(config['trainer']['percent']))
    if not os.path.isdir(file_root):
        os.mkdir(file_root)

    resume_path = parse.resume_path
    base_model = getattr(module_arch, config["arch"]['type'])()
    checkpoint = torch.load(resume_path)
    state_dict = checkpoint['state_dict']
    base_model.load_state_dict(state_dict)

    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size= 100,
        shuffle=config['data_loader']['args']['shuffle'],
        validation_split=0.0,
        num_batches=config['data_loader']['args']['num_batches'],
        training=True,
        num_workers=config['data_loader']['args']['num_workers'],
        pin_memory=config['data_loader']['args']['pin_memory'],
        config=config
    )

    if hasattr(data_loader.dataset, 'num_raw_example'):
        num_examp = data_loader.dataset.num_raw_example
    else:
        num_examp = len(data_loader.dataset)

    critenrion = nn.CrossEntropyLoss()
    
    model = Represent(base_model)
    
    isNoisy_list = np.empty((0,))
    isFalse_list = np.empty((0,))
    label_list = np.empty((0,))
    gt_list = np.empty((0,))
    conf_list = np.empty((0,))
    loss_list = np.empty((0,))
    
    #CLK / SAME
    
    model.eval()
    model.cuda()
    loss = 0

    with tqdm(data_loader) as progress:
        for batch_idx, (data, label, index, label_gt) in enumerate(progress):
            data = data.cuda()
            label, label_gt = label.long().cuda(), label_gt.long().cuda()
            output = model(data)
            _,prediction = base_model(data)
            loss = torch.nn.CrossEntropyLoss(reduction='none')(prediction, label)
            confidence, _ = torch.max(torch.nn.functional.softmax(prediction, dim=1), dim=1)
            isNoisy = label != label_gt

            gt_list = np.concatenate((gt_list, label_gt.cpu()))
            label_list = np.concatenate((label_list, label.cpu()))
            isNoisy_list = np.concatenate((isNoisy_list, isNoisy.cpu()))
            conf_list = np.concatenate((conf_list, confidence.detach().cpu()))
            loss_list = np.concatenate((loss_list, loss.detach().cpu()))
            if batch_idx == 0:
                out_list = output.detach().cpu()
            else:
                out_list = np.concatenate((out_list, output.detach().cpu()), axis=0)
                
    singular_dict, v_ortho_dict = get_singular_value_vector(label_list, out_list)

    for key in v_ortho_dict.keys():
        v_ortho_dict[key] = v_ortho_dict[key].cuda()

    sing_lbl, sin_score_lbl = singular_label(v_ortho_dict, out_list, label_list)
    kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(loss_list.reshape(-1,1))

    k_mean_cluster0 = np.mean(loss_list[kmeans.labels_ == 0])
    k_mean_cluster1 = np.mean(loss_list[kmeans.labels_ == 1])

    k_compare_label = kmeans.labels_ if k_mean_cluster0 < k_mean_cluster1 else (1 - kmeans.labels_)

    k_mean_stat = return_statistics(isNoisy_list, k_compare_label) # Selected samples, precision, recall, specificity, accuracy, fraction of clean samples
    stat_summary(config, 'CLK', k_mean_stat)
    
    fnaes_stat = return_statistics(isNoisy_list, sing_lbl)
    stat_summary(config, 'SAME', fnaes_stat)
    
    ###MCD
    #get raw data
    for batch_idx, (data, target, index, label_gt) in enumerate(data_loader):
        data, target, label_gt = data.cuda(), target.cuda(), label_gt.cuda()
        if batch_idx == 0:
            total_data = data
            total_target = target
            total_label_gt = label_gt
        else:
            total_data = torch.cat((total_data, data), 0)
            total_target = torch.cat((total_target, target), 0)
            total_label_gt = torch.cat((total_label_gt, label_gt), 0)
            
    model.eval()
    with torch.no_grad():
        temp_x = torch.rand(2,3,32,32).cuda()
#     temp_x = Variable(temp_x, volatile=True)
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list) # Number of layers that extracts feature
    total_final_feature = [0]*num_output #Extracted Features
    total = 0
    batch_size = 100

    for data_index in range(int(np.floor(total_data.size(0)/batch_size))):
        with torch.no_grad():
            data = total_data[total : total + batch_size]
#         data = Variable(data, volatile=True)

        _, out_features = model.feature_list(data)
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)
            if total == 0:
                total_final_feature[i] = out_features[i].cpu().clone()
            else:
                total_final_feature[i] = torch.cat((total_final_feature[i], out_features[i].cpu().clone()), 0)
        total += batch_size
        
    print('Random Sample Mean')
    sample_mean_list, sample_precision_list = [], []
    total_label_list = [total_target for i in range(num_output)]

    for index in range(num_output):
        sample_mean, sample_precision, _ = random_sample_mean(total_final_feature[index].cuda(), total_label_list[index].cuda(), config['num_classes'])
        sample_mean_list.append(sample_mean)
        sample_precision_list.append(sample_precision)
        
    print('Single MCD and merge the parameters')
    new_sample_mean_list = []
    new_sample_precision_list = []
    selected_feature = []
    layer_selected_index = []
    for index in range(num_output):
        tmp_selected_idx = []

        new_sample_mean = torch.Tensor(config['num_classes'], total_final_feature[index].size(1)).fill_(0).cuda()
        new_covariance = 0
        for i in range(config['num_classes']):
            index_list = total_label_list[index].eq(i) # index corresponding to each class [50000]
            temp_feature = total_final_feature[index][index_list.nonzero(), :] # feature corresponding to index_list [4972,512]
            tmp_idx_list = index_list.nonzero().view(-1).detach().cpu() # original index number of selected feature [4972]
            print(temp_feature.shape)
            temp_feature = temp_feature.view(temp_feature.size(0), -1)
            temp_mean, temp_cov, tmp_idx = MCD_single(temp_feature.cuda(), sample_mean_list[index][i], sample_precision_list[index]) # tmp_idx = MCD에서 뽑은거 [3480]
            print('selcted index for class', i, ':', tmp_idx.shape)
            new_sample_mean[i].copy_(temp_mean)
            tmp_real_idx = tmp_idx_list[tmp_idx.detach().cpu()]
            tmp_selected_idx.extend(tmp_real_idx.tolist())

            if i  == 0:
                new_covariance = temp_feature.size(0)*temp_cov
            else:
                new_covariance += temp_feature.size(0)*temp_cov

        layer_selected_index.append(tmp_selected_idx)

        new_covariance = new_covariance / total_final_feature[index].size(0)
        new_precision = scipy.linalg.pinvh(new_covariance)
        new_precision = torch.from_numpy(new_precision).float().cuda()
        new_sample_mean_list.append(new_sample_mean)
        new_sample_precision_list.append(new_precision)

    G_soft_list = []
    target_mean = new_sample_mean_list 
    target_precision = new_sample_precision_list
    
    for i in range(0, len(layer_selected_index)):
        print('# of samples in layer{}: {} '.format(i, len(layer_selected_index[i])))
                   
    print('Check for Same index')
    for i in range(len(layer_selected_index)):
        print(len([item for item, count in collections.Counter(layer_selected_index[i]).items() if count > 1]))

    #Save output_feature, target_noise, label_gt
    for i in range(num_output):
        file_name_data = '%s/%s_feature_4_%s.npy' % (file_root, data_name, str(i))
        total_feature = total_final_feature[i].numpy()
        np.save(file_name_data , total_feature)

    file_name_label = '%s/%s_target_noise.npy' % (file_root, data_name)
    np.save(file_name_label, total_target.detach().cpu())

    file_name_gt = '%s/%s_label_gt.npy' % (file_root, data_name)
    np.save(file_name_gt, total_label_gt.detach().cpu())
                   
    #Generate predicted noise index(Unselected)

    # layer_seelcted_index[0]  4_0 layer에서 mahalanobis 기준 뽑힌거
    total_index = [i for i in range(len(total_target))]
    len(total_index)
    total_index[-1]

    predicted_noise_layer = [] # Unselcted 0.3 중에서 noise인지 아닌지를 나타내는 것
    layer_unselected_index = []

    for layer in layer_selected_index:
        tmpp = set(total_index) - set(layer)
        layer_unselected_index.append(list(tmpp))

    print('Total # of dataset check: {}'.format(len(layer_unselected_index[0]) + len(layer_selected_index[0])))
    print('Tatal # of dataset check: {}'.format(len(layer_unselected_index[1]) + len(layer_selected_index[1])))
    print('Total # of dataset check: {}'.format(len(layer_unselected_index[2]) + len(layer_selected_index[2])))

    for layer in layer_unselected_index:
        tmpp_noisy = []
        num_noisy = 0
        for i in layer:
            if total_target[i] != total_label_gt[i]:
                num_noisy += 1 
                tmpp_noisy.append(1)
            else:
                tmpp_noisy.append(0)
        predicted_noise_layer.append(tmpp_noisy)
                   
    flag = 0
    predicted_clean_layer = []

    for layer in layer_selected_index:
        tmp_noisy = []
        num_noisy = 0
        for i in layer:
            if total_target[i] != total_label_gt[i]:
                num_noisy +=1 
                tmp_noisy.append(1)
            else:
                tmp_noisy.append(0)
        print('layer4_{} \nSelected samples by Mahalanobis distance: {} \nFraction of clean samples/selected samples: {}'
              .format(flag, len(layer), 1-(num_noisy/len(layer))))
        print(num_noisy)
        flag += 1
        predicted_clean_layer.append(tmp_noisy)
                   
    # Positive class = Clean / Negative Class = Noise
    recall, specificity, precision, accuracy, frac_clean, sel_samples = [], [], [], [], [], []

    for i in range(len(predicted_clean_layer)):

        tn = sum(predicted_noise_layer[i])
        fp = sum(predicted_clean_layer[i])
        tp = len(predicted_clean_layer[i]) - fp
        fn = len(predicted_noise_layer[i]) - tn

        frac_clean.append( round(tp / (tp + fp), 5))
        recall.append(round(tp / (tp + fn), 5))
        precision.append(round(tp / (tp + fp), 5))
        specificity.append(round(tn / (tn + fp), 5))
        accuracy.append(round((tp + tn) / (tp + tn + fp + fn), 5))
        sel_samples.append(tp + fp)

    # Save as txt
    df = pd.DataFrame(columns = ['MCD 0', 'MCD 1', 'MCD 2'])

    df.loc[len(df)] = sel_samples
    df.loc[len(df)] = precision
    df.loc[len(df)] = recall
    df.loc[len(df)] = specificity
    df.loc[len(df)] = accuracy
    df.loc[len(df)] = frac_clean
    df.insert(0, 'Metric', ['Sel Samples', 'Precision', 'Recall', 'Specificity', 'Accuracy', 'Fraction'])
    df['CLK'] = k_mean_stat
    df['SAME'] = fnaes_stat

    print(df)
    print(file_root)
    df.to_csv(file_root+'/metric.txt', index=False, header=True, sep="\t")
                   
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='1', type=str,
                      help='indices of GPUs to enable (default: all)')
    
    args.add_argument('--no_wandb', action='store_false', help='if false, not to use wandb')
    args.add_argument('--distillation', help='whether to distill knowledge', action='store_true')
    args.add_argument('--resume_path', type=str, default=None, help='Model for statistical analysis')
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
        CustomArgs(['--beta', '--beta'], type=float, target=('train_loss', 'args', 'beta')),
        CustomArgs(['--lambda', '--lambda'], type=float, target=('train_loss', 'args', 'lambda')),
        CustomArgs(['--percent', '--percent'], type=float, target=('trainer', 'percent')),
        CustomArgs(['--asym', '--asym'], type=bool, target=('trainer', 'asym')),
        CustomArgs(['--name', '--exp_name'], type=str, target=('name',)),
        CustomArgs(['--malpha', '--mixup_alpha'], type=float, target=('mixup_alpha',)),
        CustomArgs(['--ealpha', '--ema_alpha'], type=float, target=('ema_alpha',)),
        CustomArgs(['--nb', '--num_batches'], type=float, target=('data_loader', 'args', 'num_batches')),
        CustomArgs(['--warm', '--warmup'], type=int, target=('trainer', 'warmup')),
        CustomArgs(['--seed', '--seed'], type=int, target=('seed',)),
        CustomArgs(['--wc1', '--weight_decay1'], type=float, target=('optimizer1','weight_decay')),
        CustomArgs(['--wc2', '--weight_decay2'], type=float, target=('optimizer2','weight_decay')),
        CustomArgs(['--estep', '--ema_step'], type=float, target=('ema_step',)),

    ]
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
                   
    main(parse, config)
