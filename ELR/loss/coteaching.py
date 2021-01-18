import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

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
        
        # decay of R(T) affects co-teaching
        # forget_rate : noise rate와 동일하게 설정.
        #               noise rate는 실제로 알 수 없지만 validation set을 보면 알 수 있다고 했음.
        # num_gradual : forget rate를 유지할 것인지!
        
        self.rate_schedule = self.generate_forget_rate(forget_rate, num_gradual, n_epoch)
        
    def forward(self, logits_1, logits_2, targets, epoch, index=None, step=None):
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
        
        if len(ind_1_update) == 0:
            ind_1_update = ind_1_sorted
            ind_2_update = ind_2_sorted
            num_remember = ind_1_update
        
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
        
class CoteachingPlusLoss(CoteachingLoss):
    def __init__(self, forget_rate, num_gradual, n_epoch):
        super(CoteachingPlusLoss, self).__init__(forget_rate, num_gradual, n_epoch)
    
    def forward(self, logits, logits2, labels, epoch, ind, step):
        
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

        # TODO : step이 뭐하는 변수인지 한 번 봐야하는데..!
        # 학습 초반에 다른 애들 안거를라고 하는 건가?
        # step이 5000보다 작으면 학습 초반이라는 얘기. loss에 넣는다는 얘기
        # step이 5000보다 큰데, logical_disagree_id가 False이면 loss에 안넣는다. -> 
        # step이 5000보다 큰데, logical_disagree_id가 True이면 loss에 넣는다. ->
        
        # 그런데 disagree_id의 length가 0이면
        # disagree_id가 0이면 두 개의 network의 예측값이 모두 동일하다는 얘기
        # 그 경우면 logical_disagree가 다 False일 거고 결국 
        # update step의 여부는 step에 의해서 결정이 된다.
        # 결과적으로 step이 5000보다 큰데 disagree_id가 0이면 아무것도 학습 안시키겠다는 얘기?
        
        _update_step = np.logical_or(logical_disagree_id, step < 5000).astype(np.float32)
        update_step = Variable(torch.from_numpy(_update_step)).cuda()
        
        if len(disagree_id) > 0:
            update_labels = labels[disagree_id]
            update_outputs = logits[disagree_id] 
            update_outputs2 = logits2[disagree_id]

            loss_1, loss_2 = super().forward(update_outputs, update_outputs2, update_labels, epoch)
            
        else:
            update_labels = labels
            update_outputs = logits
            update_outputs2 = logits2

            cross_entropy_1 = F.cross_entropy(update_outputs, update_labels, reduction='none')
            cross_entropy_2 = F.cross_entropy(update_outputs2, update_labels, reduction='none')

            loss_1 = torch.sum(update_step*cross_entropy_1)/labels.size()[0]
            loss_2 = torch.sum(update_step*cross_entropy_2)/labels.size()[0]
            
        return loss_1, loss_2
    
# TODO: clean label 걸러진 애들 받아서
# Coteaching에서 CE로 학습 안시키고 H 키우는 방향으로 학습하게 하는 것.
# 아니면 아예 포함을 안시키거나.
# 태현이형한테 물어봐서 distill할 때 clean instance 어떻게 찾는지 물어볼 것.

# TODO (2021-01-18)
# 지금 non-filtered들에 대해서 entropy를 크게 하는 방향으로 계산을 하면
# Loss 레벨에서 Nan이 뜨게 됨.
# CE를 작게 하는 방향으로 학습을 하면 Nan 안뜨고 학습이 되긴 함.

class CoteachingDistillLoss(CoteachingLoss):
    def __init__(self, forget_rate, num_gradual, n_epoch, num_examp, clean_indexs):
        super(CoteachingDistillLoss, self).__init__(forget_rate, num_gradual, n_epoch)
        
        # 이거는 train_coteaching.py 레벨에서 만드는 것으로 하자!
        self.h_loss = EntropyLoss()
        self.A = math.exp(-4)
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
        remember_rate = 1 - self.rate_schedule[epoch]
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
            num_remember = ind_1_update
            
        nonfiltered_1_update = (1 - filtered_1_update.long()).bool()
        nonfiltered_2_update = (1 - filtered_2_update.long()).bool()
            
        # exchange
#         logits = torch.clamp(logits, min=self.A)
#         logits2 = torch.clamp(logits2, min=self.A)
        ce_loss_1_update = F.cross_entropy(logits[ind_2_update], labels[ind_2_update], reduction='none')
        ce_loss_2_update = F.cross_entropy(logits2[ind_1_update], labels[ind_1_update], reduction='none')
#         h_loss_1_update = self.h_loss(logits[ind_2_update])
#         h_loss_2_update = self.h_loss(logits2[ind_1_update])

#         ce_loss_1_update = F.cross_entropy(logits[filtered_2_update], labels[filtered_2_update], reduction='none')
#         ce_loss_2_update = F.cross_entropy(logits2[filtered_1_update], labels[filtered_1_update], reduction='none')
#         h_loss_1_update = self.h_loss(logits[nonfiltered_2_update])
#         h_loss_2_update = self.h_loss(logits2[nonfiltered_1_update])
        
#         loss_1_update = torch.sum(ce_loss_1_update) - torch.sum(h_loss_1_update)
#         loss_2_update = torch.sum(ce_loss_2_update) - torch.sum(h_loss_2_update)
        
#         loss_1_update = torch.sum(ce_loss_1_update[filtered_2_update]) - torch.sum(ce_loss_1_update[nonfiltered_2_update])
#         loss_2_update = torch.sum(ce_loss_2_update[filtered_1_update]) - torch.sum(ce_loss_2_update[nonfiltered_1_update])

        loss_1_update = torch.sum(ce_loss_1_update[filtered_2_update])
        loss_2_update = torch.sum(ce_loss_2_update[filtered_1_update])
    
#         loss_1_update = torch.sum(ce_loss_1_update[filtered_2_update]) - 0.3 * torch.sum(h_loss_1_update[nonfiltered_2_update])
#         loss_2_update = torch.sum(ce_loss_2_update[filtered_1_update]) - 0.3 * torch.sum(h_loss_2_update[nonfiltered_1_update])
        
        loss_1_update = loss_1_update / num_remember
        loss_2_update = loss_2_update / num_remember
        
        return loss_1_update, loss_2_update
    
class CoteachingDistillPlusLoss(CoteachingLoss):
    def __init__(self, forget_rate, num_gradual, n_epoch, clean_indx):
        super(CoteachingDistillPlusLoss, self).__init__(forget_rate, num_gradual, n_epoch)
        
    def forward(self, logits, logits2, labels, epoch):
        pass