import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class CoteachingLoss(nn.Module):
    def __init__(self, forget_rate, num_gradual, n_epoch):
        super(CoteachingLoss, self).__init__()
        
        # decay of R(T) affects co-teaching
        # forget_rate : noise rate와 동일하게 설정.
        #               noise rate는 실제로 알 수 없지만 validation set을 보면 알 수 있다고 했음.
        # num_gradual : forget rate를 유지할 것인지!
        
        self.rate_schedule = self.generate_forget_rate(forget_rate, num_gradual, n_epoch)
        
    def forward(self, logits_1, logits_2, targets, epoch, index=None):
        
        # model 1
        loss_1 = F.cross_entropy(logits_1, targets, reduction='none')
        ind_1_sorted = torch.argsort(loss_1.data).cuda()
        loss_1_sorted = loss_1[ind_1_sorted]
        
        # model 2
        loss_2 = F.cross_entropy(logits_2, targets, reduction='none')
        ind_2_sorted = torch.argsort(loss_2.data).cuda()
        loss_2_sorted = loss_2[ind_2_sorted]
        
        # sample small loss instances
        remember_rate = 1 - self.rate_schedule[epoch]
        num_remember =  int(remember_rate * len(loss_1_sorted))
        ind_1_update = ind_1_sorted[:num_remember]
        ind_2_update = ind_2_sorted[:num_remember]
        
        # exchange
        loss_1_update = F.cross_entropy(logits_1[ind_2_update], targets[ind_2_update])
        loss_2_update = F.cross_entropy(logits_2[ind_1_update], targets[ind_1_update])
        
        return loss_1_update, loss_2_update
        
        
    def generate_forget_rate(self, forget_rate, num_gradual, n_epoch):
        
        # TODO: UPDATE FORGET RATE FOR EVERY EPOCH
        # Based on Coteaching+ code

        rate_schedule = np.ones(n_epoch)*forget_rate
        rate_schedule[:num_gradual] = np.linspace(0, forget_rate, num_gradual)
        
        return rate_schedule
        
def CoteachingPlusLoss(CoteachingLoss):
    def __init__(self, forget_rate, num_gradual, n_epoch):
#         super(CoteachingPlusLoss, self).__init__(noise_rate, num_gradual, exponent, tau)
        super(CoteachingPlusLoss, self).__init__(forget_rate, num_gradual, n_epoch)
    
    def forward(self, logit, logit2, labels, ind):
        
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

        # TODO : step이 뭐하는 함수인지 한 번 봐야하는데..!    
        
        _update_step = np.logical_or(logical_disagree_id, step < 5000).astype(np.float32)
        update_step = Variable(torch.from_numpy(_update_step)).cuda()
        
        if len(disagree_id) > 0:
            update_labels = labels[disagree_id]
            update_outputs = logits[disagree_id] 
            update_outputs2 = logits2[disagree_id] 

            loss_1, loss_2 = super().forward(update_outputs, update_outputs2, update_labels)
            
        else:
            update_labels = labels
            update_outputs = logits
            update_outputs2 = logits2

            cross_entropy_1 = F.cross_entropy(update_outputs, update_labels)
            cross_entropy_2 = F.cross_entropy(update_outputs2, update_labels)

            loss_1 = torch.sum(update_step*cross_entropy_1)/labels.size()[0]
            loss_2 = torch.sum(update_step*cross_entropy_2)/labels.size()[0]

#             pure_ratio_1 = np.sum(noise_or_not[ind])/ind.shape[0]
#             pure_ratio_2 = np.sum(noise_or_not[ind])/ind.shape[0]
        return loss_1, loss_2
        
    
        
        