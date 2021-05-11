# Code for Discard Noisy Instance Dynamically
# With Gaussian Mixture Model

import numpy as np
import math
from sklearn.mixture import GaussianMixture as GMM
import scipy.stats as stats
import torch

from .svd_classifier import get_singular_value_vector
from .svd_classifier import singular_label
from .svd_classifier import kmean_singular_label

def same_score(v_ortho_dict, features, labels):
    features = torch.from_numpy(features).cuda()
    scores = torch.zeros(features.shape[0])
    
    for indx, feat in enumerate(features):
        scores[indx] = torch.dot(v_ortho_dict[labels[indx]][0], feat).abs()
    return scores

def same_mixture_model(label_list, scores):
    
    output = []
    for idx in range(len(np.unique(label_list))):
        indexs= torch.tensor(range(len(scores)))[label_list==idx]
        feats = scores[label_list==idx].cpu().numpy()
        feats_ = np.ravel(feats).astype(np.float).reshape(-1, 1)
        g = GMM(n_components=2, covariance_type='full', tol=1e-6, max_iter=1000)
        
        g.fit(feats_)
        weights, means, covars = g.weights_, g.means_, g.covariances_
        
        # boundary? QDA!
        a = (1/2) * ((1/covars[0]) - (1/covars[1]))
        b = -(means[0]/covars[0]) + (means[1]/covars[1])
        c = (1/2) * ((np.square(means[0])/covars[0]) - (np.square(means[1])/covars[1]))
        c -= np.log((weights[0])/np.sqrt(2*np.pi*covars[0]))
        c += np.log((weights[1])/np.sqrt(2*np.pi*covars[1]))
        d = b**2 - 4*a*c
        
#         if d > 0:
#             bound = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
#             if bound > min(means) and bound < max(means):
#                 num_instance = len(indexs[feats > bound]) * 0.9
#                 f = feats_.copy().ravel()
#                 f.sort()
#                 bound = f[-int(num_instance)]
                
#                 output += indexs[feats > bound].numpy().tolist()
#             else:
#                 bound = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
#                 num_instance = len(indexs[feats > bound]) * 0.9
#                 f = feats_.copy().ravel()
#                 f.sort()
                
#                 print (len(f), num_instance)
                
#                 bound = f[-int(num_instance)]
                
#                 output += indexs[feats > bound].numpy().tolist()
            
#         else:
#             clean = 0 if means[0] > means[1] else 1
            
#             f = feats_.copy().ravel()
#             f.sort()
#             num_instance = len(f) * (weights[clean])
#             bound = f[-int(num_instance)]
#             output += indexs[feats > bound].numpy().tolist()
        
        bound = estimate_purity(feats, means, covars, weights)
        output += indexs[feats > bound].numpy().tolist()
    
    return torch.tensor(output).long()

def same_mixture_index(orig_label, orig_out, prev_label, prev_out):
    singular_dict, v_ortho_dict = get_singular_value_vector(prev_label, prev_out)
    for key in v_ortho_dict.keys():
        v_ortho_dict[key] = v_ortho_dict[key].cuda()
        
    scores = same_score(v_ortho_dict, orig_out, orig_label)
    output = same_mixture_model(orig_label, scores)
    return output.numpy()

def estimate_purity(f, means, covars, weights):
    
    best_f1 = 0
    for x in np.linspace(min(means), max(means), 100):
        x0 = (x - means[0]) / np.sqrt(covars[0])
        x1 = (x - means[1]) / np.sqrt(covars[1])

        cdf0 = 1 - stats.norm.cdf(x0)
        cdf1 = 1 - stats.norm.cdf(x1)

        if means[0] > means[1]:
            pred_purity = (weights[0]*cdf0) / (weights[0]*cdf0 + weights[1]*cdf1)
            c_instances = weights[0] * len(f)
            c = weights[0]
        else:
            pred_purity = (weights[1]*cdf1) / (weights[1]*cdf1 + weights[0]*cdf0)
            c_instances = weights[1] * len(f)
            c = weights[1]
            
            
        precision = pred_purity
        recall = pred_purity * len(f[f > x]) / c_instances
        f1 = 2 * precision * recall / (precision + recall)
        
        if recall > 0.8 and precision > best_f1:
            best_f1 = precision
            boundary = x
            
    print ('Clean: {}'.format(c))
    return boundary
    

# def same_topk(label_list, scores, p):
    
#     output = []
#     for idx in range(len(np.unique(label_list))):
#         num_inst = int(p * np.sum(label_list==idx))
#         indexs = torch.tensor(range(50000))[label_list==idx]
#         tmp_sort, tmp_idx = torch.sort(scores[label_list==idx], descending=False)
#         # 못 들어간 애가 필요한거니까 이렇게!
#         output += indexs[tmp_idx[num_inst:]].numpy().tolist()
        
#     return torch.tensor(output).long()

# def same_kmeans(label_list, scores, p=None):
    
#     output = []
#     for idx in range(len(np.unique(label_list))):
#         indexs = torch.tensor(range(len(scores)))[label_list==idx]
#         kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(scores[indexs].reshape(-1, 1))
        
#         if torch.mean(scores[indexs][kmeans.labels_==0]) < torch.mean(scores[indexs][kmeans.labels_==1]):
#             kmeans.labels_ = 1 - kmeans.labels_
#         output += indexs[kmeans.labels_ == 0].numpy().tolist()
        
#     return torch.tensor(output).long()
        

# def same_topk_index(orig_label_list, orig_out_list, prev_label_list, prev_out_list, p=None):
    
#     singular_dict, v_ortho_dict = get_singular_value_vector(prev_label_list, prev_out_list)
#     for key in v_ortho_dict.keys():
#         v_ortho_dict[key] = v_ortho_dict[key].cuda()
        
#     scores = same_score(v_ortho_dict, orig_out_list, orig_label_list)
#     output = same_topk(orig_label_list, scores, p)
#     return output.numpy()

# def same_kmeans_index(orig_label_list, orig_out_list, prev_label_list, prev_out_list, p=None):
    
#     singular_dict, v_ortho_dict = get_singular_value_vector(prev_label_list, prev_out_list)
#     for key in v_ortho_dict.keys():
#         v_ortho_dict[key] = v_ortho_dict[key].cuda()
        
#     scores = same_score(v_ortho_dict, orig_out_list, orig_label_list)
#     output = same_kmeans(orig_label_list, scores, p)
#     return output.numpy()
    
# def compute_noisy_ratio(data_loader):
#     isNoisy_list = np.empty((0,))
    
#     with tqdm(data_loader) as progress:
#         for _, (_, label, index, label_gt) in enumerate(progress):
#             isNoisy = label == label_gt
#             isNoisy_list = np.concatenate((isNoisy_list, isNoisy.cpu()))

#     print ('#############################')
#     print (isNoisy_list.sum(), isNoisy_list.shape)
#     print('purity in this dataset: {}'.format(isNoisy_list.sum() / isNoisy_list.shape))
    