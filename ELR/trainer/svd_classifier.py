import torch
import numpy as np
from tqdm import tqdm
from sklearn import cluster

def get_loss_list(model, data_loader):
    loss_list = np.empty((0,))
    
    with tqdm(data_loader) as progress:
        for batch_idx, (data, label, index, label_gt) in enumerate(progress):
            data = data.cuda()
            label, label_gt = label.long().cuda(), label_gt.long().cuda()

            _, prediction = model(data)
            loss = torch.nn.CrossEntropyLoss(reduction='none')(prediction, label)

            loss_list = np.concatenate((loss_list, loss.detach().cpu()))
    
    kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(loss_list.reshape(-1,1))
    
    if np.mean(loss_list[kmeans.labels_==0]) > np.mean(loss_list[kmeans.labels_==1]):
        clean_label = 1
    else:
        clean_label = 0
    
    output=[]
    for idx, value in enumerate(kmeans.labels_):
        if value==clean_label:
            output.append(idx)
    
    return output

def get_loss_list_2d(model, data_loader, n_clusters=2):
    loss_list = np.empty((0, 2))
    model.cuda()
    
    with tqdm(data_loader) as progress:
        for batch_idx, (data, label, index, label_gt) in enumerate(progress):
            data = data.cuda()
            label, label_gt = label.long().cuda(), label_gt.long().cuda()

            _, pred = model(data)
            loss = torch.nn.CrossEntropyLoss(reduction='none')(pred, label)
            
            prob = torch.softmax(pred, dim=-1)
            
            top2_log_pred, top2_ind = torch.topk(torch.log(prob), k=2, dim=-1)
            is_pred_wrong = (top2_ind[:, 0] != label).bool()
            is_pred_correct = (top2_ind[:, 0] == label).bool()
            
            label_top1 = torch.stack([loss, -top2_log_pred[:, 0]], dim=1) # for pred wrong
            top2_log_pred = -top2_log_pred
            top2_log_pred[is_pred_wrong] = label_top1[is_pred_wrong]

            loss_list = np.concatenate((loss_list, top2_log_pred.detach().cpu().numpy()), axis=0)
    
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=0).fit(loss_list.reshape(50000,2))
    
    mean_losses = []
    for itr in range(n_clusters):
        mean_losses.append(np.mean(loss_list[kmeans.labels_==itr][:, 0]))
    
    _, clean_label = torch.topk(-torch.tensor(mean_losses), k=1)
    
    output=[]
    for idx, value in enumerate(kmeans.labels_):
        if value==clean_label:
            output.append(idx)
    
    return output

def iterative_eigen(number, label_list, out_list):
    sin_lbls = {}
    
    for i in range(number):
        tmp_lbl = torch.zeros(50000)
        for k in range(i):
            tmp_lbl += sin_lbls[k] 
        singular_dict, v_ortho_dict = get_singular_value_vector(label_list[tmp_lbl==0], out_list[tmp_lbl==0])

        for key in v_ortho_dict.keys():
            v_ortho_dict[key] = v_ortho_dict[key].cuda()

        sing_lbl, sin_score_lbl = singular_label(v_ortho_dict, out_list, label_list)
        sin_lbls[i]=sing_lbl
        if i>0 and torch.all(torch.eq(sin_lbls[i], sin_lbls[i-1])):
            print(i)
            break
            
    output=[]
    for idx, value in enumerate(sing_lbl):
        if value==0:
            output.append(idx)
        
    return output

    
def get_out_list(model, data_loader):

    label_list = np.empty((0,))

    model.eval()
    model.cuda()
    with tqdm(data_loader) as progress:
        for batch_idx, (data, label, index, label_gt) in enumerate(progress):
            data = data.cuda()
            label, label_gt = label.long().cuda(), label_gt.long().cuda()
            output, _ = model(data)

            label_list = np.concatenate((label_list, label.cpu()))
            if batch_idx == 0:
                out_list = output.detach().cpu()
            else:
                out_list = np.concatenate((out_list, output.detach().cpu()), axis=0)
    
    return label_list, out_list


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

def isNoisy_ratio(data_loader):
    isNoisy_list = np.empty((0,))
    with tqdm(data_loader) as progress:
        for batch_idx, (data, label, index, label_gt) in enumerate(progress):
            data = data.cuda()
            isNoisy = label != label_gt

            isNoisy_list = np.concatenate((isNoisy_list, isNoisy.cpu()))

    
    print('purity in this dataset: {}'.format(isNoisy_list.sum() / isNoisy_list.shape))
    
