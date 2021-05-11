import torch
import numpy as np
from tqdm import tqdm
from sklearn import cluster

#bol_norm True -> Divide by norm of feature
def same_score(v_ortho_dict, features, labels, bol_norm=False):
    features = torch.from_numpy(features).cuda()
    scores = torch.zeros(features.shape[0])
    
    for indx, feat in enumerate(features):
        tmp_scores = torch.dot(v_ortho_dict[labels[indx]][0], feat).abs() 
        scores[indx] = (tmp_scores / torch.norm(feat, p=2)) if bol_norm else tmp_scores
    return scores

def same_topk(label_list, scores, p):
    
    output = []
    for idx in range(len(np.unique(label_list))):
        num_inst = int(p * np.sum(label_list==idx))
        indexs = torch.tensor(range(len(label_list)))[label_list==idx]
        tmp_sort, tmp_idx = torch.sort(scores[label_list==idx], descending=False)
        # 못 들어간 애가 필요한거니까 이렇게!
        output += indexs[tmp_idx[num_inst:]].numpy().tolist()
        
    return torch.tensor(output).long()

#Classswise kmenas
def same_kmeans(label_list, scores, p=None):
    
    output = []
    for idx in range(len(np.unique(label_list))):
        indexs = torch.tensor(range(len(scores)))[label_list==idx]
        kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(scores[indexs].reshape(-1, 1))
        
        if torch.mean(scores[indexs][kmeans.labels_==0]) < torch.mean(scores[indexs][kmeans.labels_==1]):
            kmeans.labels_ = 1 - kmeans.labels_
        output += indexs[kmeans.labels_ == 0].numpy().tolist()
        
    return torch.tensor(output).long()
        
#Total Kmeans
def same_kmeans_total(scores, p=None):
    output = []
    indexs = torch.tensor(range(len(scores)))
    kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(scores.reshape(-1, 1))
    
    if torch.mean(scores[kmeans.labels_==0]) < torch.mean(scores[kmeans.labels_==1]):
        kmeans.labels_ = 1 - kmeans.labels_
    
    for idx, value in enumerate(kmeans.labels_):
        if value == 0:
            output.append(idx)
    
    return torch.tensor(output).long(), None

def same_topk_index(orig_label_list, orig_out_list, prev_label_list, prev_out_list, p=None):
    
    singular_dict, v_ortho_dict = get_singular_value_vector(prev_label_list, prev_out_list)
    for key in v_ortho_dict.keys():
        v_ortho_dict[key] = v_ortho_dict[key].cuda()
        
    scores = same_score(v_ortho_dict, orig_out_list, orig_label_list)
    output = same_topk(orig_label_list, scores, p)
    return output.numpy()

def same_kmeans_index(orig_label_list, orig_out_list, prev_label_list, prev_out_list, p=None):
    
    singular_dict, v_ortho_dict = get_singular_value_vector(prev_label_list, prev_out_list)
    for key in v_ortho_dict.keys():
        v_ortho_dict[key] = v_ortho_dict[key].cuda()
        
    scores = same_score(v_ortho_dict, orig_out_list, orig_label_list)
    output = same_kmeans(orig_label_list, scores, p)
    return output.numpy()
    
def compute_noisy_ratio(data_loader):
    isNoisy_list = np.empty((0,))
    
    with tqdm(data_loader) as progress:
        for _, (_, label, index, label_gt) in enumerate(progress):
            isNoisy = label == label_gt
            isNoisy_list = np.concatenate((isNoisy_list, isNoisy.cpu()))

    print ('#############################')
    print (isNoisy_list.sum(), isNoisy_list.shape)
    print('purity in this dataset: {}'.format(isNoisy_list.sum() / isNoisy_list.shape))


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

def iterative_eigen(number, label_list, out_list, teacher_idx=None):
    sin_lbls = {}
    for i in range(number):
        tmp_lbl = torch.zeros(50000)
        if teacher_idx !=None:
            for num in (set(range(0,50000)) - set(teacher_idx)):
                tmp_lbl[num] += 1
        print(tmp_lbl.sum().item())
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
    if number ==1:
        output=[]
        for idx, value in enumerate(sing_lbl):
            if value==0:
                output.append(idx)
    else:
        kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(loss_list.reshape(-1,1))
    
        if np.mean(sin_score_lbl[kmeans.labels_==0]) > np.mean(sin_score_lbl[kmeans.labels_==1]):
            clean_label = 0
        else:
            clean_label = 1
        
        output=[]
        for idx, value in enumerate(kmeans.labels_):
            if value==clean_label:
                output.append(idx)
        
        
    return output

    
def get_out_list(model, data_loader):

    label_list = np.empty((0,))

    model.eval()
    model.cuda()
    with tqdm(data_loader) as progress:
        for batch_idx, (data, label, index, _) in enumerate(progress):
            data = data.cuda()
#             label, label_gt = label.long().cuda(), label_gt.long().cuda()
            label = label.long()
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

def kmean_singular_label(v_ortho_dict, model_represents, label):
    
    model_represents = torch.from_numpy(model_represents).cuda()
    sing_lbl = torch.zeros(model_represents.shape[0])
    sin_score_lbl = torch.zeros(model_represents.shape[0])
    
    for i, data in enumerate(model_represents):
        sin_score_lbl[i] = torch.dot(v_ortho_dict[label[i]][0], data).abs() - torch.dot(v_ortho_dict[label[i]][1], data).abs()
        
    kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(sin_score_lbl.reshape(-1, 1))
    
    if torch.mean(sin_score_lbl[kmeans.labels_==0]) < torch.mean(sin_score_lbl[kmeans.labels_==1]):
        kmeans.labels_ = 1 - kmeans.labels_
    
    output = []
    for idx, value in enumerate(kmeans.labels_):
        if value == 0:
            output.append(idx)
    
    return output

def kmean_singular_label2(v_ortho_dict, model_represents, label):
    
    model_represents = torch.from_numpy(model_represents).cuda()
    sing_lbl = torch.zeros(model_represents.shape[0])
    sin_score_lbl = torch.zeros(model_represents.shape[0])
    
    for i, data in enumerate(model_represents):
        sin_score_lbl[i] = torch.dot(v_ortho_dict[label[i]][0], data).abs() / torch.norm(data, p=2)
        
    kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(sin_score_lbl.reshape(-1, 1))
    
    if torch.mean(sin_score_lbl[kmeans.labels_==0]) < torch.mean(sin_score_lbl[kmeans.labels_==1]):
        kmeans.labels_ = 1 - kmeans.labels_
    
    output = []
    for idx, value in enumerate(kmeans.labels_):
        if value == 0:
            output.append(idx)
    
    return output

def kmean_eigen_out(label_list, out_list, teacher_idx=None):
    singular_dict, v_ortho_dict = get_singular_value_vector(label_list, out_list)
    
    for key in v_ortho_dict.keys():
        v_ortho_dict[key] = v_ortho_dict[key].cuda()
    
    output = kmean_singular_label(v_ortho_dict, out_list, label_list)
    
    return output

def topk_eigen_kmean(label_list, out_list, teacher_idx=None):
    singular_dict, v_ortho_dict = get_singular_value_vector(label_list, out_list)
    
    for key in v_ortho_dict.keys():
        v_ortho_dict[key] = v_ortho_dict[key].cuda()
    
    output = kmean_singular_label2(v_ortho_dict, out_list, label_list)
    
    return output

def get_anchor(label_list, out_list, teacher_idx=None):
    
    label_list = torch.from_numpy(label_list).long().numpy()
    singular_dict, v_ortho_dict = get_singular_value_vector(label_list, out_list)
    
    for key in v_ortho_dict.keys():
        v_ortho_dict[key] = v_ortho_dict[key].cuda()
    
    model_represents = torch.from_numpy(out_list).cuda()
    sin_score_lbl = [[] for _ in range(len(np.unique(label_list)))]
    
    for i, data in enumerate(model_represents):
        sin_score_lbl[label_list[i]].append(torch.dot(v_ortho_dict[label_list[i]][0], data).abs())
    
    # classwise topk
    v_ortho_dict_ = {}
    for index in np.unique(label_list):
        cls_score_lbl = sin_score_lbl[index]
        topk_v, topk_i = torch.topk(torch.tensor(cls_score_lbl), k=50)
        
        u, s, v = np.linalg.svd(model_represents[label_list==index][topk_i].cpu().numpy())
        v_ortho_dict_[index] = torch.from_numpy(v[0]).unsqueeze(0).cuda()
        
    output = kmean_singular_label2(v_ortho_dict_, model_represents.cpu().numpy(), label_list)
    return output
        

def isNoisy_ratio(data_loader):
    isNoisy_list = np.empty((0,))
    with tqdm(data_loader) as progress:
        for _, (_, label, index, label_gt) in enumerate(progress):
            isNoisy = label == label_gt
            isNoisy_list = np.concatenate((isNoisy_list, isNoisy.cpu()))

    print ('#############################')
    print (isNoisy_list.sum(), isNoisy_list.shape)
    print('purity in this dataset: {}'.format(isNoisy_list.sum() / isNoisy_list.shape))

    
def extract_teacherIdx(teacher, data_loader, parse):
    teacher.load_state_dict(torch.load('./checkpoint/' + parse.load_name)['state_dict'])
    teacher = teacher.cuda()
    if not parse.reinit:
        model.load_state_dict(torch.load('./checkpoint/' + parse.load_name)['state_dict'])
    for params in teacher.parameters():
        params.requires_grad = False
    if parse.distill_mode == 'eigen':
        tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
        teacher_idx = iterative_eigen(1,tea_label_list,tea_out_list)
    elif parse.distill_mode == 'fulleigen':
        tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
        teacher_idx = iterative_eigen(100,tea_label_list,tea_out_list)
    elif parse.distill_mode == 'kmean_eigen':
        tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
        teacher_idx = kmean_eigen_out(tea_label_list, tea_out_list)
    elif parse.distill_mode == 'topk_eigen_kmean':
        tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
        teacher_idx = topk_eigen_kmean(tea_label_list, tea_out_list)
    else:
        teacher_idx = get_loss_list(teacher, data_loader)
    print('||||||original||||||')
    isNoisy_ratio(data_loader)
    if parse.second_load_name !=None:
        teacher.load_state_dict(torch.load('./checkpoint/' + parse.second_load_name)['state_dict'])
        teacher = teacher.cuda()
        if not parse.reinit:
            model.load_state_dict(torch.load('./checkpoint/' + parse.second_load_name)['state_dict'])
        for params in teacher.parameters():
            params.requires_grad = False
        if parse.distill_mode == 'eigen':
            tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
            teacher_idx2 = iterative_eigen(1,tea_label_list,tea_out_list,teacher_idx)
        elif parse.distill_mode == 'fulleigen':
            tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
            teacher_idx2 = iterative_eigen(100,tea_label_list,tea_out_list)
        else:
            teacher_idx2 = get_loss_list(teacher, data_loader)
        teacher_idx = list(set(teacher_idx) & set(teacher_idx2))
        print('second_distillation')
        if parse.third_load_name !=None:
            teacher.load_state_dict(torch.load('./checkpoint/' + parse.third_load_name)['state_dict'])
            teacher = teacher.cuda()
            if not parse.reinit:
                model.load_state_dict(torch.load('./checkpoint/' + parse.third_load_name)['state_dict'])
            for params in teacher.parameters():
                params.requires_grad = False
            if parse.distill_mode == 'eigen':
                tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
                teacher_idx3 = iterative_eigen(1,tea_label_list,tea_out_list, teacher_idx)
            elif parse.distill_mode == 'fulleigen':
                tea_label_list, tea_out_list = get_out_list(teacher, data_loader)
                teacher_idx3 = iterative_eigen(100,tea_label_list,tea_out_list)
            else:
                teacher_idx3 = get_loss_list(teacher, data_loader)
            teacher_idx = list(set(teacher_idx) & set(teacher_idx3))
            print('third_ distillation')

    return teacher_idx


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
