import argparse
import torch
import sys
import os
import json
import random
import numpy as np
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

#Base Resnet
class Represent(nn.Module):
    def __init__(self, base_model):
        super(Represent, self).__init__()
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.linear = base_model.linear
        
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        y = out.view(out.size(0), -1)
        
        return y
    
    #Feature Extractting
    def feature_list(self, x):
        output_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        for name, module in self.layer1._modules.items():
            out = module(out)
        for name, module in self.layer2._modules.items():
            out = module(out)
        for name, module in self.layer3._modules.items():
            out = module(out)
        for name, module in self.layer4._modules.items():
            out = module(out)
            output_list.append(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y, output_list
    
def get_singular_value_vector(label_list, out_list):
    
    singular_dict = {}
    v_ortho_dict = {}
    
    for index in np.unique(label_list):
        u, s, v = np.linalg.svd(out_list[label_list==index])
        singular_dict[index] = s[0] / s[1]
        v_ortho_dict[index] = torch.from_numpy(v[:2])

    return singular_dict, v_ortho_dict

def singular_label(v_ortho_dict, model_represents, label):
    
    model_represents = torch.from_numpy(model_represents).cuda()
    sing_lbl = torch.zeros(model_represents.shape[0]) 
    sin_score_lbl = torch.zeros(model_represents.shape[0])
    
    for i, data in enumerate(model_represents):
        sin_score_lbl[i] = torch.dot(v_ortho_dict[label[i]][0], data).abs() - torch.dot(v_ortho_dict[label[i]][1], data).abs()
        if torch.dot(v_ortho_dict[label[i]][0], data).abs() < torch.dot(v_ortho_dict[label[i]][1], data).abs():
            sing_lbl[i] = 1
        
    return sing_lbl, sin_score_lbl

def return_statistics(isNoisy_list, predict):
    r_stats = []
    
    tp = (isNoisy_list[predict==0]==0).sum()
    tn = isNoisy_list[predict==1].sum()
    fp = isNoisy_list.sum() - tn
    fn = ((isNoisy_list.shape - isNoisy_list.sum()) - tp).item()
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sel_samples = int(fp + tp)
    frac_clean = tp / (fp + tp)

    r_stats.extend([sel_samples, round(precision, 5), round(recall, 5), round(specificity, 5), round(accuracy, 5), round(frac_clean, 5)])
    
    return r_stats

def stat_summary(config, name, stat_list):
    print('Dataset: {}, Net: {}, Noise{}_{}, Loss: {}'
      .format(config['data_loader']['type'], config['arch']['type'], config['trainer']['asym'], config['trainer']['percent'], config['train_loss']['type']))
    
    print("="*50, name , "="*50)

    print('Selected samples by {}: {} \nPrecision: {} \nRecall: {} \nSpecificity: {}\nAccuracy: {} \nFraction of clean samples/selected samples: {}'
                  .format(name, stat_list[0], stat_list[1], stat_list[2], stat_list[3], stat_list[4], stat_list[5]))
    
def report_metric(config, sel_samples, precision, recall, specificity, accuracy, frac_clean):
    print('Dataset: {}, Net: {}, Noise{}_{}, Loss: {}'
      .format(config['data_loader']['type'], config['arch']['type'], config['trainer']['asym'], config['trainer']['percent'], config['train_loss']['type']))
    print("="*50, 'Mahalanobis Distance', "="*50)
    
    if len(recall) > 1:
        for i in range(len(recall)):
            print('layer4_{} \nSelected samples by Mahalanobis distance: {} \nPrecision: {} \nRecall: {} \nSpecificity: {}\nAccuracy: {} \nFraction of clean samples/selected samples: {}'
                  .format(i, sel_samples[i], precision[i], recall[i], specificity[i], accuracy[i], frac_clean[i]))
    else:
        print('layer4_{} \nSelected samples by Mahalanobis distance: {} \nPrecision: {} \nRecall: {} \nSpecificity: {}\nAccuracy: {} \nFraction of clean samples/selected samples: {}'
                  .format(sel_samples, precision, recall, specificity, accuracy, frac_clean))
    
#Random Sample Mean
def random_sample_mean(feature, total_label, num_classes):
    
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered = False)    
    new_feature, fraction_list = [], []
    frac = 0.7
    sample_mean_per_class = torch.Tensor(num_classes, feature.size(1)).fill_(0).cuda()
    total_label = total_label.cuda()

    
    total_selected_list = []
    for i in range(num_classes):
        index_list = total_label.eq(i)
        temp_feature = feature[index_list.nonzero(), :]
        temp_feature = temp_feature.view(temp_feature.size(0), -1)
        shuffler_idx = torch.randperm(temp_feature.size(0))
        index = shuffler_idx[:int(temp_feature.size(0)*frac)]
        fraction_list.append(int(temp_feature.size(0)*frac))
        total_selected_list.append(index_list.nonzero()[index.cuda()])

        selected_feature = torch.index_select(temp_feature, 0, index.cuda())
        new_feature.append(selected_feature)
        sample_mean_per_class[i].copy_(torch.mean(selected_feature, 0))
    
    total_covariance = 0
    for i in range(num_classes):
        flag = 0
        X = 0
        for j in range(fraction_list[i]):
            temp_feature = new_feature[i][j]
            temp_feature = temp_feature - sample_mean_per_class[i]
            temp_feature = temp_feature.view(-1,1)
            if flag  == 0:
                X = temp_feature.transpose(0,1)
                flag = 1
            else:
                X = torch.cat((X,temp_feature.transpose(0,1)),0)
            # find inverse            
        group_lasso.fit(X.cpu().numpy())
        inv_sample_conv = group_lasso.covariance_
        inv_sample_conv = torch.from_numpy(inv_sample_conv).float().cuda()
        if i == 0:
            total_covariance = inv_sample_conv*fraction_list[i]
        else:
            total_covariance += inv_sample_conv*fraction_list[i]
        total_covariance = total_covariance/sum(fraction_list)
    new_precision = scipy.linalg.pinvh(total_covariance.cpu().numpy())
    new_precision = torch.from_numpy(new_precision).float().cuda()
    
    return sample_mean_per_class, new_precision, total_selected_list

#MCD single
def MCD_single(feature, sample_mean, inverse_covariance):
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    temp_batch = 100
    total, mahalanobis_score = 0, 0
    frac = 0.7 #fraction N_c+d+1 / 2
    for data_index in range(int(np.ceil(feature.size(0)/temp_batch))):
        temp_feature = feature[total : total + temp_batch].cuda()        
        gaussian_score = 0
        batch_sample_mean = sample_mean
        zero_f = temp_feature - batch_sample_mean
        term_gau = -0.5*torch.mm(torch.mm(zero_f, inverse_covariance), zero_f.t()).diag()
        # concat data
        if total == 0:
            mahalanobis_score = term_gau.view(-1,1)
        else:
            mahalanobis_score = torch.cat((mahalanobis_score, term_gau.view(-1,1)), 0)
        total += temp_batch
        
    mahalanobis_score = mahalanobis_score.view(-1)
    feature = feature.view(feature.size(0), -1)
    _, selected_idx = torch.topk(mahalanobis_score, int(feature.size(0)*frac)) # 전체 class에서 selected index
    selected_feature = torch.index_select(feature, 0, selected_idx.cuda())
    new_sample_mean = torch.mean(selected_feature, 0)
    
    # compute covariance matrix
    X = 0
    flag = 0
    for j in range(selected_feature.size(0)):
        temp_feature = selected_feature[j]
        temp_feature = temp_feature - new_sample_mean
        temp_feature = temp_feature.view(-1,1)
        if flag  == 0:
            X = temp_feature.transpose(0,1)
            flag = 1
        else:
            X = torch.cat((X, temp_feature.transpose(0,1)),0)
    # find inverse            
    group_lasso.fit(X.cpu().numpy())
    new_sample_cov = group_lasso.covariance_
    
    return new_sample_mean, new_sample_cov, selected_idx