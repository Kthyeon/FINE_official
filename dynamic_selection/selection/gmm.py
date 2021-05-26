# Code for Discard Noisy Instance Dynamically
# With Gaussian Mixture Model

import numpy as np
import math
import scipy.stats as stats
import torch

from sklearn.mixture import GaussianMixture as GMM
# from .svd_classifier import get_singular_vector, cleansing, get_score
from .util import estimate_purity

__all__=['fit_mixture', 'fit_mixture_bmm']


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
        gmm = GMM(n_components=2, covariance_type='full', tol=1e-6, max_iter=100)
        
        gmm.fit(feats_)
        prob = gmm.predict_proba(feats_)
        prob = prob[:,gmm.means_.argmax()]
        clean_labels += [cls_index[clean_idx] for clean_idx in range(len(cls_index)) if prob[clean_idx] > p_threshold] 
    
    return np.array(clean_labels, dtype=np.int64)

def fit_mixture_bmm(scores, labels, p_threshold=0.5):
    """
    Assum the distribution of scores: bimodal beta mixture model
    
    return clean labels
    that belongs to the clean cluster by fitting the score distribution to BMM
    """
    
    clean_labels = []
    indexes = np.array(range(len(scores)))
    for cls in np.unique(labels):
        cls_index = indexes[labels==cls]
        feats = scores[labels==cls]
        feats_ = np.ravel(feats).astype(np.float).reshape(-1, 1)
        feats_ = (feats_ - feats_.min()) / (feats_.max() - feats_.min())
        bmm = BetaMixture(max_iters=100)
        bmm.fit(feats_)
        
        mean_0 = bmm.alphas[0] / (bmm.alphas[0] + bmm.betas[0])
        mean_1 = bmm.alphas[1] / (bmm.alphas[1] + bmm.betas[1])
        clean = 0 if mean_0 > mean_1 else 1
        
        init = bmm.predict(feats_.min(), p_threshold, clean)
        for x in np.linspace(feats_.min(), feats_.max(), 50):
            pred = bmm.predict(x, p_threshold, clean)
            if pred != init:
                bound = x
                break
        
        clean_labels += [cls_index[clean_idx] for clean_idx in range(len(cls_index)) if feats[clean_idx] > bound] 
    
    return np.array(clean_labels, dtype=np.int64)


################### CODE FOR THE BETA MODEL  ###################

def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)

def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) /x_bar
    return alpha, beta

class BetaMixture(object):
    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12
        
    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])
    
    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)
    
    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))
    
    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)
    
    def responsibilities(self, x):
        r =  np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r
    
    def score_samples(self, x):
        return -np.log(self.probability(x))
    
    def fit(self, x):
        x = np.copy(x)

        # EM on beta distributions unsable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):

            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

        return self
    
    def predict(self, x, threshold, clean):
        return self.posterior(x, clean) > threshold
    
    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l # I do not use this one at the end
        
    def look_lookup(self, x, loss_max, loss_min):
        x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]
    
    def plot(self):
        x = np.linspace(0, 1, 100)
        plt.plot(x, self.weighted_likelihood(x, 0), label='negative')
        plt.plot(x, self.weighted_likelihood(x, 1), label='positive')
        plt.plot(x, self.probability(x), lw=2, label='mixture')
        
        
    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)


def fine_gmm(current_features, current_labels, prev_features=None, prev_labels=None):
    '''
    prev_features, prev_labels: data from the previous round
    current_features, current_labels: current round's data
    
    return clean labels
    
    if you insert the prev_features and prev_labels to None,
    the algorthm divides the data based on the current labels and current features
    
    '''
    
    if (prev_features != None) and (prev_labels != None):
        singular_vector_dict = get_singular_vector(prev_features, prev_labels)
    else:
        singular_vector_dict = get_singular_vector(current_features, current_labels)

        
    scores = get_score(singular_vector_dict, current_features, current_labels)
    output = fit_mixture(orig_label, scores)
    return output.numpy()


    

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
    