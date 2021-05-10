import torch
import numpy as np
from tqdm import tqdm
from sklearn import cluster
from tqdm import tqdm
from .gmm import *
from .util import *

__all__=['get_singular_vector', 'cleansing', 'fine', 'extract_cleanidx']


def get_singular_vector(features, labels):
    '''
    To get top1 sigular vector in class-wise manner by using SVD of hidden feature vectors
    features: hidden feature vectors of data (numpy)
    labels: correspoding label list
    '''
    
    singular_vector_dict = {}
    with tqdm(total=len(np.unique(labels))) as pbar:
        for index in np.unique(labels):
            _, _, v = np.linalg.svd(features[labels==index])
            singular_vector_dict[index] = v[0]
            pbar.update(1)

    return singular_vector_dict


def get_score(singular_vector_dict, features, labels):
    '''
    Calculate the score providing the degree of showing whether the data is clean or not.
    '''
    scores = [np.abs(np.inner(singular_vector_dict[labels[indx]], feat/torch.norm(feat))) for indx, feat in enumerate(tqdm(features))]
    return np.array(scores)

def extract_topk(scores, labels, k):
    '''
    k: ratio to extract topk scores in class-wise manner
    To obtain the most prominsing clean data in each classes
    
    return selected labels 
    which contains top k data
    '''
    
    indexes = torch.tensor(range(len(labels)))
    selected_labels = []
    for cls in np.unique(labels):
        num = int(p * np.sum(labels==cls))
        _, sorted_idx = torch.sort(scores[labels==cls], descending=True)
        selected_labels += indexes[labels==cls][sorted_idx[:num]].numpy().tolist()
        
    return torch.tensor(selected_labels, dtype=torch.int64)

def cleansing(scores, labels):
    '''
    Assume the distribution of scores: bimodal spherical distribution.
    
    return clean labels 
    that belongs to the clean cluster made by the KMeans algorithm
    '''
    
    indexes = np.array(range(len(scores)))
    clean_labels = []
    for cls in np.unique(labels):
        cls_index = indexes[labels==cls]
        kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(scores[cls_index].reshape(-1, 1))
        if np.mean(scores[cls_index][kmeans.labels_==0]) < np.mean(scores[cls_index][kmeans.labels_==1]): kmeans.labels_ = 1 - kmeans.labels_
            
        clean_labels += cls_index[kmeans.labels_ == 0].tolist()
        
    return np.array(clean_labels, dtype=np.int64)
        

def fine(current_features, current_labels, fit = 'kmeans', prev_features=None, prev_labels=None):
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
        
    scores = get_score(singular_vector_dict, features = current_features, labels = current_labels)
    if fit == 'kmeans':
        clean_labels = cleansing(scores, current_labels)
    elif fit == 'gmm':
        clean_labels = fit_mixture(scores, current_labels)
    else:
        raise NotImplemented
    
    return clean_labels

def extract_cleanidx(teacher, data_loader, parse, print_statistics = True):
    teacher.load_state_dict(torch.load('./checkpoint/' + parse.load_name)['state_dict'])
    teacher = teacher.cuda()

    if not parse.reinit: teacher.load_state_dict(torch.load('./checkpoint/' + parse.load_name)['state_dict'])
    for params in teacher.parameters(): params.requires_grad = False
    
    features, labels = get_features(teacher, data_loader)
    clean_labels = fine(current_features=features, current_labels=labels, fit = parse.distill_mode)
    
    if print_statistics: return_statistics(data_loader, clean_labels, datanum=len(labels))
    
    return clean_labels
    

# def extract_teacherIdx(teacher, data_loader, parse):
#     teacher.load_state_dict(torch.load('./checkpoint/' + parse.load_name)['state_dict'])
#     teacher = teacher.cuda()
#     if not parse.reinit:
#         model.load_state_dict(torch.load('./checkpoint/' + parse.load_name)['state_dict'])
#     for params in teacher.parameters():
#         params.requires_grad = False
#     if parse.distill_mode == 'eigen':
#         tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
#         teacher_idx = iterative_eigen(1,tea_label_list,tea_out_list)
#     elif parse.distill_mode == 'fulleigen':
#         tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
#         teacher_idx = iterative_eigen(100,tea_label_list,tea_out_list)
#     elif parse.distill_mode == 'kmean_eigen':
#         tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
#         teacher_idx = kmean_eigen_out(tea_label_list, tea_out_list)
#     elif parse.distill_mode == 'topk_eigen_kmean':
#         tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
#         teacher_idx = topk_eigen_kmean(tea_label_list, tea_out_list)
#     else:
#         teacher_idx = get_loss_list(teacher, data_loader)
#     print('||||||original||||||')
#     isNoisy_ratio(data_loader)
#     if parse.second_load_name !=None:
#         teacher.load_state_dict(torch.load('./checkpoint/' + parse.second_load_name)['state_dict'])
#         teacher = teacher.cuda()
#         if not parse.reinit:
#             model.load_state_dict(torch.load('./checkpoint/' + parse.second_load_name)['state_dict'])
#         for params in teacher.parameters():
#             params.requires_grad = False
#         if parse.distill_mode == 'eigen':
#             tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
#             teacher_idx2 = iterative_eigen(1,tea_label_list,tea_out_list,teacher_idx)
#         elif parse.distill_mode == 'fulleigen':
#             tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
#             teacher_idx2 = iterative_eigen(100,tea_label_list,tea_out_list)
#         else:
#             teacher_idx2 = get_loss_list(teacher, data_loader)
#         teacher_idx = list(set(teacher_idx) & set(teacher_idx2))
#         print('second_distillation')
#         if parse.third_load_name !=None:
#             teacher.load_state_dict(torch.load('./checkpoint/' + parse.third_load_name)['state_dict'])
#             teacher = teacher.cuda()
#             if not parse.reinit:
#                 model.load_state_dict(torch.load('./checkpoint/' + parse.third_load_name)['state_dict'])
#             for params in teacher.parameters():
#                 params.requires_grad = False
#             if parse.distill_mode == 'eigen':
#                 tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
#                 teacher_idx3 = iterative_eigen(1,tea_label_list,tea_out_list, teacher_idx)
#             elif parse.distill_mode == 'fulleigen':
#                 tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
#                 teacher_idx3 = iterative_eigen(100,tea_label_list,tea_out_list)
#             else:
#                 teacher_idx3 = get_loss_list(teacher, data_loader)
#             teacher_idx = list(set(teacher_idx) & set(teacher_idx3))
#             print('third_ distillation')

#     return teacher_idx




# def iterative_eigen(number, label_list, out_list, teacher_idx=None):
#     sin_lbls = {}
#     for i in range(number):
#         tmp_lbl = torch.zeros(50000)
#         if teacher_idx !=None:
#             for num in (set(range(0,50000)) - set(teacher_idx)):
#                 tmp_lbl[num] += 1
#         print(tmp_lbl.sum().item())
#         for k in range(i):
#             tmp_lbl += sin_lbls[k] 
#         singular_dict, v_ortho_dict = get_singular_value_vector(label_list[tmp_lbl==0], out_list[tmp_lbl==0])

#         for key in v_ortho_dict.keys():
#             v_ortho_dict[key] = v_ortho_dict[key].cuda()

#         sing_lbl, sin_score_lbl = singular_label(v_ortho_dict, out_list, label_list)
#         sin_lbls[i]=sing_lbl
#         if i>0 and torch.all(torch.eq(sin_lbls[i], sin_lbls[i-1])):
#             print(i)
#             break
#     if number ==1:
#         output=[]
#         for idx, value in enumerate(sing_lbl):
#             if value==0:
#                 output.append(idx)
#     else:
#         kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(loss_list.reshape(-1,1))
    
#         if np.mean(sin_score_lbl[kmeans.labels_==0]) > np.mean(sin_score_lbl[kmeans.labels_==1]):
#             clean_label = 0
#         else:
#             clean_label = 1
        
#         output=[]
#         for idx, value in enumerate(kmeans.labels_):
#             if value==clean_label:
#                 output.append(idx)
        
        
#     return output

    





# def singular_label(v_ortho_dict, model_represents, label):
    
#     model_represents = torch.from_numpy(model_represents).cuda()
#     sing_lbl = torch.zeros(model_represents.shape[0]) 
#     sin_score_lbl = torch.zeros(model_represents.shape[0])
    
#     for i, data in enumerate(model_represents):
#         sin_score_lbl[i] = torch.dot(v_ortho_dict[label[i]][0], data).abs() - torch.dot(v_ortho_dict[label[i]][1], data).abs()
#         if torch.dot(v_ortho_dict[label[i]][0], data).abs() < torch.dot(v_ortho_dict[label[i]][1], data).abs():
#             sing_lbl[i] = 1
        
#     return sing_lbl, sin_score_lbl

# def kmean_singular_label(v_ortho_dict, model_represents, label):
    
#     model_represents = torch.from_numpy(model_represents).cuda()
#     sing_lbl = torch.zeros(model_represents.shape[0])
#     sin_score_lbl = torch.zeros(model_represents.shape[0])
    
#     for i, data in enumerate(model_represents):
#         sin_score_lbl[i] = torch.dot(v_ortho_dict[label[i]][0], data).abs() - torch.dot(v_ortho_dict[label[i]][1], data).abs()
        
#     kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(sin_score_lbl.reshape(-1, 1))
    
#     if torch.mean(sin_score_lbl[kmeans.labels_==0]) < torch.mean(sin_score_lbl[kmeans.labels_==1]):
#         kmeans.labels_ = 1 - kmeans.labels_
    
#     output = []
#     for idx, value in enumerate(kmeans.labels_):
#         if value == 0:
#             output.append(idx)
    
#     return output

# def kmean_singular_label2(v_ortho_dict, model_represents, label):
    
#     model_represents = torch.from_numpy(model_represents).cuda()
#     sing_lbl = torch.zeros(model_represents.shape[0])
#     sin_score_lbl = torch.zeros(model_represents.shape[0])
    
#     for i, data in enumerate(model_represents):
#         sin_score_lbl[i] = torch.dot(v_ortho_dict[label[i]][0], data).abs() / torch.norm(data, p=2)
        
#     kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(sin_score_lbl.reshape(-1, 1))
    
#     if torch.mean(sin_score_lbl[kmeans.labels_==0]) < torch.mean(sin_score_lbl[kmeans.labels_==1]):
#         kmeans.labels_ = 1 - kmeans.labels_
    
#     output = []
#     for idx, value in enumerate(kmeans.labels_):
#         if value == 0:
#             output.append(idx)
    
#     return output

# def kmean_eigen_out(features, labels, teacher_idx=None):
#     singular_dict, v_ortho_dict = get_singular_value_vector(label_list, out_list)
    
#     for key in v_ortho_dict.keys():
#         v_ortho_dict[key] = v_ortho_dict[key].cuda()
    
#     output = kmean_singular_label(v_ortho_dict, out_list, label_list)
    
#     return output

# def topk_eigen_kmean(label_list, out_list, teacher_idx=None):
#     singular_dict, v_ortho_dict = get_singular_value_vector(label_list, out_list)
    
#     for key in v_ortho_dict.keys():
#         v_ortho_dict[key] = v_ortho_dict[key].cuda()
    
#     output = kmean_singular_label2(v_ortho_dict, out_list, label_list)
    
#     return output


        

    
# def extract_teacherIdx(teacher, data_loader, parse):
#     teacher.load_state_dict(torch.load('./checkpoint/' + parse.load_name)['state_dict'])
#     teacher = teacher.cuda()
#     if not parse.reinit:
#         model.load_state_dict(torch.load('./checkpoint/' + parse.load_name)['state_dict'])
#     for params in teacher.parameters():
#         params.requires_grad = False
#     if parse.distill_mode == 'eigen':
#         tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
#         teacher_idx = iterative_eigen(1,tea_label_list,tea_out_list)
#     elif parse.distill_mode == 'fulleigen':
#         tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
#         teacher_idx = iterative_eigen(100,tea_label_list,tea_out_list)
#     elif parse.distill_mode == 'kmean_eigen':
#         tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
#         teacher_idx = kmean_eigen_out(tea_label_list, tea_out_list)
#     elif parse.distill_mode == 'topk_eigen_kmean':
#         tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
#         teacher_idx = topk_eigen_kmean(tea_label_list, tea_out_list)
#     else:
#         teacher_idx = get_loss_list(teacher, data_loader)
#     print('||||||original||||||')
#     isNoisy_ratio(data_loader)
#     if parse.second_load_name !=None:
#         teacher.load_state_dict(torch.load('./checkpoint/' + parse.second_load_name)['state_dict'])
#         teacher = teacher.cuda()
#         if not parse.reinit:
#             model.load_state_dict(torch.load('./checkpoint/' + parse.second_load_name)['state_dict'])
#         for params in teacher.parameters():
#             params.requires_grad = False
#         if parse.distill_mode == 'eigen':
#             tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
#             teacher_idx2 = iterative_eigen(1,tea_label_list,tea_out_list,teacher_idx)
#         elif parse.distill_mode == 'fulleigen':
#             tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
#             teacher_idx2 = iterative_eigen(100,tea_label_list,tea_out_list)
#         else:
#             teacher_idx2 = get_loss_list(teacher, data_loader)
#         teacher_idx = list(set(teacher_idx) & set(teacher_idx2))
#         print('second_distillation')
#         if parse.third_load_name !=None:
#             teacher.load_state_dict(torch.load('./checkpoint/' + parse.third_load_name)['state_dict'])
#             teacher = teacher.cuda()
#             if not parse.reinit:
#                 model.load_state_dict(torch.load('./checkpoint/' + parse.third_load_name)['state_dict'])
#             for params in teacher.parameters():
#                 params.requires_grad = False
#             if parse.distill_mode == 'eigen':
#                 tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
#                 teacher_idx3 = iterative_eigen(1,tea_label_list,tea_out_list, teacher_idx)
#             elif parse.distill_mode == 'fulleigen':
#                 tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
#                 teacher_idx3 = iterative_eigen(100,tea_label_list,tea_out_list)
#             else:
#                 teacher_idx3 = get_loss_list(teacher, data_loader)
#             teacher_idx = list(set(teacher_idx) & set(teacher_idx3))
#             print('third_ distillation')

#     return teacher_idx


# def get_loss_list_2d(model, data_loader, n_clusters=2, c_clusters=1):
#     loss_list = np.empty((0, 2))
#     model.cuda()
    
#     with tqdm(data_loader) as progress:
#         for batch_idx, (data, label, index, label_gt) in enumerate(progress):
#             data = data.cuda()
#             label, label_gt = label.long().cuda(), label_gt.long().cuda()

#             _, pred = model(data)
#             loss = torch.nn.CrossEntropyLoss(reduction='none')(pred, label)
            
#             prob = torch.softmax(pred, dim=-1)
            
#             top2_log_pred, top2_ind = torch.topk(torch.log(prob), k=n_clusters, dim=-1)
#             is_pred_wrong = (top2_ind[:, 0] != label).bool()
#             is_pred_correct = (top2_ind[:, 0] == label).bool()
            
#             label_top1 = torch.stack([loss, -top2_log_pred[:, 0]], dim=1) # for pred wrong
#             top2_log_pred = -top2_log_pred
#             top2_log_pred[is_pred_wrong] = label_top1[is_pred_wrong]

#             loss_list = np.concatenate((loss_list, top2_log_pred.detach().cpu().numpy()), axis=0)
    
#     kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=0).fit(loss_list.reshape(50000,2))
    
#     mean_losses = []
#     for itr in range(n_clusters):
#         mean_losses.append(np.mean(loss_list[kmeans.labels_==itr][:, 0]))
    
#     _, clean_labels = torch.topk(-torch.tensor(mean_losses), k=c_clusters)
    
#     output=[]
#     for idx, value in enumerate(kmeans.labels_):
#         if value in clean_labels:
#             output.append(idx)
    
#     return output





# def same_topk_index(orig_label_list, orig_out_list, prev_label_list, prev_out_list, p=None):
    
#     singular_dict, v_ortho_dict = get_singular_value_vector(prev_label_list, prev_out_list)
#     for key in v_ortho_dict.keys():
#         v_ortho_dict[key] = v_ortho_dict[key].cuda()
        
#     scores = same_score(v_ortho_dict, orig_out_list, orig_label_list)
#     output = same_topk(orig_label_list, scores, p)
#     return output.numpy()