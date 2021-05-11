# Code for Discard Noisy Instance Dynamically
# With Gaussian Mixture Model

import numpy as np
import math
import scipy.stats as stats
import torch

from sklearn.mixture import GaussianMixture as GMM
from .svd_classifier import get_singular_vector, cleansing, get_score
from .util import estimate_purity

__all__=['fit_mixture']


def fit_mixture(scores, labels, p_threshold=0.5):
    '''
    Assume the distribution of scores: bimodal gaussian mixture model
    
    return clean labels
    that belongs to the clean cluster by fitting the score distribution to GMM
    '''
    
    clean_labels = []
    indexes = np.array(range(len(scores)))
    for cls in np.unique(labels):
        cls_index = indexes[labels==cls]
        feats = scores[labels==cls]
        feats_ = np.ravel(feats).astype(np.float).reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, covariance_type='full', tol=1e-6, max_iter=10)
        
        gmm.fit(feats_)
        prob = gmm.predict_proba(feats_)
        prob = prob[:,gmm.means_.argmax()]         
#         weights, means, covars = g.weights_, g.means_, g.covariances_
        
#         # boundary? QDA!
#         a, b = (1/2) * ((1/covars[0]) - (1/covars[1])), -(means[0]/covars[0]) + (means[1]/covars[1])
#         c = (1/2) * ((np.square(means[0])/covars[0]) - (np.square(means[1])/covars[1]))
#         c -= np.log((weights[0])/np.sqrt(2*np.pi*covars[0]))
#         c += np.log((weights[1])/np.sqrt(2*np.pi*covars[1]))
#         d = b**2 - 4*a*c
        
#         bound = estimate_purity(feats, means, covars, weights)
        clean_labels += [cls_index[clean_idx] for clean_idx in range(len(cls_index)) if prob[clean_idx] > p_threshold] 
    
    return np.array(clean_labels, dtype=np.int64)    



# def fine_gmm(current_features, current_labels, prev_features=None, prev_labels=None):
#     '''
#     prev_features, prev_labels: data from the previous round
#     current_features, current_labels: current round's data
    
#     return clean labels
    
#     if you insert the prev_features and prev_labels to None,
#     the algorthm divides the data based on the current labels and current features
    
#     '''
    
#     if (prev_features != None) and (prev_labels != None):
#         singular_vector_dict = get_singular_vector(prev_features, prev_labels)
#     else:
#         singular_vector_dict = get_singular_vector(current_features, current_labels)

        
#     scores = get_score(singular_vector_dict, current_features, current_labels)
#     output = fit_mixture(orig_label, scores)
#     return output.numpy()


    

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
    