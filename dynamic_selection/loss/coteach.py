import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math


__all__=['CoteachingLoss', 'CoteachingPlusLoss', 'CoteachingDistillLoss']

class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()
        
    def forward(self, logits):
        pred = F.softmax(logits, dim=1)
        loss = -torch.sum(pred * torch.log(pred), dim=1)
        return loss

class CoteachingLoss(nn.Module):
    def __init__(self, forget_rate, num_gradual, n_epoch):
        super(CoteachingLoss, self).__init__()
        
        self.rate_schedule = self.generate_forget_rate(forget_rate, num_gradual, n_epoch)
        
    def forward(self, logits_1, logits_2, targets, gt, epoch, index=None, step=None):
        
        # model 1
        loss_1 = F.cross_entropy(logits_1, targets, reduction='none')
        ind_1_sorted = torch.argsort(loss_1.data).cuda()
        loss_1_sorted = loss_1[ind_1_sorted]
        
        # model 2
        loss_2 = F.cross_entropy(logits_2, targets, reduction='none')
        ind_2_sorted = torch.argsort(loss_2.data).cuda()
        loss_2_sorted = loss_2[ind_2_sorted]
        
        # sample small loss instances
        remember_rate = 1 - self.rate_schedule[epoch-1]
        num_remember =  int(remember_rate * len(loss_1_sorted))
        
        ind_1_update = ind_1_sorted[:num_remember]
        ind_2_update = ind_2_sorted[:num_remember]
        
        if len(ind_1_update) == 0:
            ind_1_update = ind_1_sorted
            ind_2_update = ind_2_sorted
            num_remember = ind_1_update
        
        # exchange
        loss_1_update = F.cross_entropy(logits_1[ind_2_update], targets[ind_2_update])
        loss_2_update = F.cross_entropy(logits_2[ind_1_update], targets[ind_1_update])
        
        clean1, total1 = (targets[ind_2_update] == gt[ind_2_update]).sum(), len(targets[ind_2_update] == gt[ind_2_update])
        clean2, total2 = (targets[ind_1_update] == gt[ind_1_update]).sum(), len(targets[ind_1_update] == gt[ind_1_update])
        
        return loss_1_update, loss_2_update, clean1, clean2, total1, total2
        
        
    def generate_forget_rate(self, forget_rate, num_gradual, n_epoch):
        
        # TODO: UPDATE FORGET RATE FOR EVERY EPOCH
        # Based on Coteaching+ code
        forget_rate *= 0.9
        rate_schedule = np.ones(n_epoch)*forget_rate
        rate_schedule[:num_gradual-1] = np.linspace(0, forget_rate, num_gradual-1)
        
        return rate_schedule
        
class CoteachingPlusLoss(CoteachingLoss):
    def __init__(self, forget_rate, num_gradual, n_epoch):
        super(CoteachingPlusLoss, self).__init__(forget_rate, num_gradual, n_epoch)
    
    def forward(self, logits, logits2, labels, gt, epoch, ind, step):
        
        ind = ind.cpu().numpy().transpose()
        
        # disagreement
        _, pred1 = torch.max(logits.data, 1)
        _, pred2 = torch.max(logits2.data, 1)
        
        pred1, pred2 = pred1.cpu().numpy(), pred2.cpu().numpy()
        
        logical_disagree_id=np.zeros(labels.size(), dtype=bool)
        disagree_id = []
        for idx, p1 in enumerate(pred1): 
            if p1 != pred2[idx]:
                disagree_id.append(idx) 
                logical_disagree_id[idx] = True

        temp_disagree = ind*logical_disagree_id.astype(np.int64)
        ind_disagree = np.asarray([i for i in temp_disagree if i != 0]).transpose()
        try:
            assert ind_disagree.shape[0]==len(disagree_id)
        except:
            disagree_id = disagree_id[:ind_disagree.shape[0]]
        
        _update_step = np.logical_or(logical_disagree_id, step < 5000).astype(np.float32)
        update_step = Variable(torch.from_numpy(_update_step)).cuda()
        
        if len(disagree_id) > 0:
            update_labels = labels[disagree_id]
            update_outputs = logits[disagree_id] 
            update_outputs2 = logits2[disagree_id]
            update_gt = gt[disagree_id]

            loss_1, loss_2, clean1, clean2, total1, total2 = super().forward(update_outputs, update_outputs2, update_labels, update_gt, epoch)
            
        else:
            update_labels = labels
            update_outputs = logits
            update_outputs2 = logits2
            update_gt = gt

            cross_entropy_1 = F.cross_entropy(update_outputs, update_labels, reduction='none')
            cross_entropy_2 = F.cross_entropy(update_outputs2, update_labels, reduction='none')

            loss_1 = torch.sum(update_step*cross_entropy_1)/labels.size(0)
            loss_2 = torch.sum(update_step*cross_entropy_2)/labels.size(0)
            
            clean1, total1 = (update_labels == update_gt).sum(), len(update_labels == update_gt)
            clean2, total2 = (update_labels == update_gt).sum(), len(update_labels == update_gt)
            
        return loss_1, loss_2, clean1, clean2, total1, total2

class CoteachingDistillLoss(CoteachingLoss):
    def __init__(self, forget_rate, num_gradual, n_epoch, num_examp, clean_indexs):
        super(CoteachingDistillLoss, self).__init__(forget_rate, num_gradual, n_epoch)
        
        # 이거는 train_coteaching.py 레벨에서 만드는 것으로 하자!
        self.h_loss = EntropyLoss()
        self.generate_filtered_array(num_examp, clean_indexs)
        
    def generate_filtered_array(self, num_examp, clean_indexs):
        self.is_in_teacher_idx = torch.Tensor([False for _ in range(num_examp)])
        self.is_in_teacher_idx[clean_indexs] = True
        
    def forward(self, logits, logits2, labels, epoch, index, step=None):
        
        # determine filtered or non-filtered
        filtered = self.is_in_teacher_idx[index].bool()
        
        # model 1
        loss_1 = F.cross_entropy(logits, labels, reduction='none')
        ind_1_sorted = torch.argsort(loss_1.data).cuda()
        loss_1_sorted = loss_1[ind_1_sorted]
        filtered_1_sorted = filtered[ind_1_sorted]
        
        # model 2
        loss_2 = F.cross_entropy(logits2, labels, reduction='none')
        ind_2_sorted = torch.argsort(loss_2.data).cuda()
        loss_2_sorted = loss_2[ind_2_sorted]
        filtered_2_sorted = filtered[ind_2_sorted]
        
        # sample small loss instances
        remember_rate = 1 - self.rate_schedule[epoch-1]
        num_remember =  int(remember_rate * len(loss_1_sorted))
        
        ind_1_update = ind_1_sorted[:num_remember]
        ind_2_update = ind_2_sorted[:num_remember]
        filtered_1_update = filtered_1_sorted[:num_remember]
        filtered_2_update = filtered_2_sorted[:num_remember]
        
        if len(ind_1_update) == 0:
            ind_1_update = ind_1_sorted
            ind_2_update = ind_2_sorted
            filtered_1_update = filtered_1_sorted
            filtered_2_update = filtered_2_sorted
            num_remember = len(ind_1_update)
            
        nonfiltered_1_update = (1 - filtered_1_update.long()).bool()
        nonfiltered_2_update = (1 - filtered_2_update.long()).bool()
            
        # exchange
        ce_loss_1_update = F.cross_entropy(logits[ind_2_update], labels[ind_2_update], reduction='none')
        ce_loss_2_update = F.cross_entropy(logits2[ind_1_update], labels[ind_1_update], reduction='none')
        
#         h_loss_1_update = self.h_loss(logits[ind_2_update])
#         h_loss_2_update = self.h_loss(logits2[ind_1_update])
        
#         loss_1_update = torch.sum(ce_loss_1_update) - torch.sum(h_loss_1_update)
#         loss_2_update = torch.sum(ce_loss_2_update) - torch.sum(h_loss_2_update)
        
#         loss_1_update = torch.sum(ce_loss_1_update[filtered_2_update]) - torch.sum(ce_loss_1_update[nonfiltered_2_update])
#         loss_2_update = torch.sum(ce_loss_2_update[filtered_1_update]) - torch.sum(ce_loss_2_update[nonfiltered_1_update])

        loss_1_update = torch.sum(ce_loss_1_update[filtered_2_update])
        loss_2_update = torch.sum(ce_loss_2_update[filtered_1_update])
        
        size_1 = num_remember if torch.sum(filtered_2_update) == 0 else torch.sum(filtered_2_update)
        size_2 = num_remember if torch.sum(filtered_1_update) == 0 else torch.sum(filtered_1_update) 
        
        loss_1_update = loss_1_update / size_1
        loss_2_update = loss_2_update / size_2
        
        return loss_1_update, loss_2_update
    
class CoteachingPlusDistillLoss(CoteachingLoss):
    def __init__(self, forget_rate, num_gradual, n_epoch, num_examp, clean_indexs):
        super(CoteachingPlusDistillLoss, self).__init__(forget_rate, num_gradual, n_epoch)
        
        self.generate_filtered_array(num_examp, clean_indexs)
        
    def generate_filtered_array(self, num_examp, clean_indexs):
        self.is_in_teacher_idx = torch.Tensor([False for _ in range(num_examp)])
        self.is_in_teacher_idx[clean_indexs] = True
        
    def forward(self, logits, logits2, labels, epoch, index, step=None):
        ind = index.cpu().numpy().transpose()
        
        # disagreement
        _, pred1 = torch.max(logits.data, 1)
        _, pred2 = torch.max(logits2.data, 1)
        
        pred1, pred2 = pred1.cpu().numpy(), pred2.cpu().numpy()
        
        logical_disagree_id=np.zeros(labels.size(), dtype=bool)
        disagree_id = []
        for idx, p1 in enumerate(pred1):
            if p1 != pred2[idx]:
                disagree_id.append(idx)
                logical_disagree_id[idx] = True

        temp_disagree = ind*logical_disagree_id.astype(np.int64)
        ind_disagree = np.asarray([i for i in temp_disagree if i != 0]).transpose()
        try:
            assert ind_disagree.shape[0]==len(disagree_id)
        except:
            disagree_id = disagree_id[:ind_disagree.shape[0]]
        
        _update_step = np.logical_or(logical_disagree_id, step < 5000).astype(np.float32)
        update_step = Variable(torch.from_numpy(_update_step)).cuda()
        
        disagree_instances = index[disagree_id]
        filtered = self.is_in_teacher_idx[disagree_instances].bool()
        disagree_id = torch.Tensor(disagree_id)[filtered].long()
        
        if len(disagree_id) > 0:
            update_labels = labels[disagree_id]
            update_outputs = logits[disagree_id] 
            update_outputs2 = logits2[disagree_id]

            loss_1, loss_2 = super().forward(update_outputs, update_outputs2, update_labels, epoch)
            
        else:
            update_labels = labels
            update_outputs = logits
            update_outputs2 = logits2
            
            filtered = self.is_in_teacher_idx[index].bool()

            cross_entropy_1 = F.cross_entropy(update_outputs, update_labels, reduction='none')[filtered]
            cross_entropy_2 = F.cross_entropy(update_outputs2, update_labels, reduction='none')[filtered]
            update_step = update_step[filtered]
            
            size = labels.size(0) if torch.sum(update_step) == 0 else torch.sum(update_step)

            loss_1 = torch.sum(update_step*cross_entropy_1) / size
            loss_2 = torch.sum(update_step*cross_entropy_2) / size
            
        return loss_1, loss_2