import torch
import numpy as np
from tqdm import tqdm
from sklearn import cluster
import torch.nn.functional as F

def get_loss_list(model, data_loader):
    loss_list = np.empty((0,))

    with tqdm(data_loader) as progress:
        for batch_idx, (data, label, index) in enumerate(progress):
            data = data.cuda()
            label = label.long().cuda()

            prediction = model(data)
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


def singular_label(v_ortho_dict, model_represents, label):
    
    sing_lbl = torch.zeros(model_represents.shape[0]) == 0.
    
    for i, data in enumerate(model_represents):
        data = torch.from_numpy(data).cuda()
        if torch.dot(v_ortho_dict[label[i].item()][0], data).abs() < torch.dot(v_ortho_dict[label[i].item()][1], data).abs():
            sing_lbl[i] = False
    
    output=[]
    for idx, value in enumerate(sing_lbl):
        if value:
            output.append(idx)
        
    return output

    
def get_out_list(model, data_loader):

    label_list = np.empty((0,))

    model.eval()
    model.cuda()
    with tqdm(data_loader) as progress:
        for batch_idx, (data, label, index) in enumerate(progress):
            data = data.cuda()
            label = label.long().cuda()
            output = model.forward(data, lout=4)
            output = F.avg_pool2d(output, 4)
            output = output.view(output.size(0), -1)

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
        singular_dict[index] = s[0] - s[1]
        v_ortho_dict[index] = torch.from_numpy(v[:2])

    return singular_dict, v_ortho_dict