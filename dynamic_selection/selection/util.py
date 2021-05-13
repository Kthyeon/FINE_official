import numpy as np
from tqdm import tqdm
import torch
import scipy.stats as stats
from sklearn import cluster

__all__=['compute_noiseratio', 'get_features', 'estimate_purity', 'return_statistics','cleansing_loss']


def compute_noiseratio(dataloader):
    '''
    get the noisy list in the current dataloader
    '''
    isNoisy_list = np.empty((0,))
    
    with tqdm(dataloader) as progress:
        for _, (_, label, _, label_gt) in enumerate(progress):
            isNoisy = label == label_gt
            isNoisy_list = np.concatenate((isNoisy_list, isNoisy.cpu()))
    
    # clean: 1, noisy 0
    return isNoisy_list
    
def return_statistics(dataloader, selected_idx):
    '''
    selected_idx: list of selected clean labels by filtering method
    ''' 
    isNoisy_list = compute_noiseratio(dataloader)
    r_stats = []
    
    tp = isNoisy_list[selected_idx].sum()
    fp = (selected_idx.shape - tp).item()
    fn = isNoisy_list.sum() - tp
    tn = (isNoisy_list==0).sum() -fp
    
    print('Real Noisy: {}, Real Clean: {}'.format((isNoisy_list==0).sum(), (isNoisy_list==1).sum()))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 / ((1 / precision) + (1  / recall))
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sel_samples = '{}/{}'.format(tp, tp+fp)

    r_stats.extend([sel_samples, round(precision, 4), round(recall, 4), round(f1_score, 4), round(specificity, 4), round(accuracy, 4)])
    print('Selected_clean/total: {} \nPrecision: {} \nRecall: {} \nF1_Score: {} \nSpecificity: {}\nAccuracy: {}'.format(r_stats[0], r_stats[1], r_stats[2], r_stats[3], r_stats[4], r_stats[5]))
    
    return r_stats[0], r_stats[1], r_stats[2], r_stats[3], r_stats[4], r_stats[5]
    
    
def get_features(model, dataloader):
    '''
    Concatenate the hidden features and corresponding labels 
    '''
    labels = np.empty((0,))

    model.eval()
    model.cuda()
    with tqdm(dataloader) as progress:
        for batch_idx, (data, label, _, _) in enumerate(progress):
            data, label = data.cuda(), label.long()
            feature, _ = model(data)

            labels = np.concatenate((labels, label.cpu()))
            if batch_idx == 0:
                features = feature.detach().cpu()
            else:
                features = np.concatenate((features, feature.detach().cpu()), axis=0)
    
    return features, labels

def cleansing_loss(model, dataloader):
    
    loss_list = np.empty((0,))
    labels = np.empty((0,))
    model.eval()
    model.cuda()
    with tqdm(dataloader) as progress:
        for batch_idx, (data, label, _, _) in enumerate(progress):
            data, label = data.cuda(), label.long().cuda()
            _, prediction = model(data)
            
            loss = torch.nn.CrossEntropyLoss(reduction='none')(prediction, label)
            
            loss_list = np.concatenate((loss_list, loss.detach().cpu()))
            labels = np.concatenate((labels, label.detach().cpu()))

    kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(loss_list.reshape(-1, 1))
    if np.mean(loss_list[kmeans.labels_==0]) > np.mean(loss_list[kmeans.labels_==1]): kmeans.labels_ = 1 - kmeans.labels_
        
    indexes = np.array(range(len(labels)))

    return np.array(indexes[kmeans.labels_ == 0], dtype=np.int64), labels

def estimate_purity(f, means, covars, weights):
    '''
    Estimate the purity of the current dataloader
    '''
    
    best_f1 = 0
    for x in np.linspace(f.min(), f.max(), 100):
        x0, x1 = (x - means[0]) / np.sqrt(covars[0]), (x - means[1]) / np.sqrt(covars[1])

        cdf0, cdf1 = 1 - stats.norm.cdf(x0), 1 - stats.norm.cdf(x1)

        if means[0] > means[1]:
            pred_purity, c_instances = (weights[0]*cdf0) / (weights[0]*cdf0 + weights[1]*cdf1), weights[0] * len(f)
            print ('Clean: {}'.format(weights[0]))
        else:
            pred_purity, c_instances = (weights[1]*cdf1) / (weights[1]*cdf1 + weights[0]*cdf0), weights[1] * len(f)
            print ('Clean: {}'.format(weights[1]))
            
        precision, recall = pred_purity, pred_purity * len(f[f > x]) / c_instances
        f1 = 2 * precision * recall / (precision + recall)
        
    if f1 > best_f1:
        boundary = x

    return boundary