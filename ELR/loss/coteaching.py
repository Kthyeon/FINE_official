import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# Loss function
# def CoteachingLoss(y_1, y_2, t, forget_rate, indexs, noise):
    
#     """
#     noise: bool, whether or not instance is noise or clean
#     이거 원래 코드에서는 불러왔는데
#     우리 코드에서는 gt랑 label이랑 비교해서 결정해야 할듯.
#     """
    
#     loss_1 = F.cross_entropy(y_1, t, reduce=False)
#     ind_1_sorted = np.argsort(loss_1.data).cuda()
#     loss_1_sorted = loss_1[ind_1_sorted]
    
#     loss_2 = F.cross_entropy(y_2, t, reduce=False)
#     ind_2_sorted = np.argsort(loss_2.data).cuda()
#     loss_2_sorted = loss_2[ind_2_sorted]
    
#     remember_rate = 1 - forget_rate
#     num_remember = int(remember_rate * len(loss_1_sorted))
    
#     # TODO: noise_or_not 계산해서 pure_ratio 계산하는 데 사용할 것.
#     # 일단은 이게 학습 시키는데 당장 필요한 요소는 아니긴 함.
# #     pure_ratio_1 = np.sum(noise[indexs[ind_1_sorted[:num_remember]]])/float(num_remember)
# #     pure_ratio_2 = np.sum(noise[indexs[ind_2_sorted[:num_remember]]])/float(num_remember)
    
#     ind_1_update = ind_1_sorted[:num_remember]
#     ind_2_update = ind_2_sorted[:num_remember]
    
#     loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
#     loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])
    
#     return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember

class CoteachingLoss(nn.Module):
    def __init__(self, noise_rate, num_gradual=10., exponent=1., tau=1.):
        super(CoteachingLoss, self).__init__()
        
        # decay of R(T) affects co-teaching
        # --num_gradual : Tk
        # --exponent : c
        # --tau : {0.5, 0.75, 1, 1.25, 1.5} * epsilon
        self.forget_rate = 0
        self.noise_rate = noise_rate
        self.num_gradual = num_gradual
        self.exponent = exponent
        self.tau = tau
        
    def forward(self, logits_1, logits_2, targets):
        
        # model 1
        loss_1 = F.cross_entropy(logits_1, targets, reduction='none')
        ind_1_sorted = torch.argsort(loss_1.data).cuda()
        loss_1_sorted = loss_1[ind_1_sorted]
        
        # model 2
        loss_2 = F.cross_entropy(logits_2, targets, reduction='none')
        ind_2_sorted = torch.argsort(loss_2.data).cuda()
        loss_2_sorted = loss_2[ind_2_sorted]
        
        # sample small loss instances
        remember_rate = 1 - self.forget_rate
        num_remember =  int(remember_rate * len(loss_1_sorted))
        ind_1_update = ind_1_sorted[:num_remember]
        ind_2_update = ind_2_sorted[:num_remember]
        
        # exchange
        loss_1_update = F.cross_entropy(logits_1[ind_2_update], targets[ind_2_update])
        loss_2_update = F.cross_entropy(logits_2[ind_1_update], targets[ind_1_update])
        
        return loss_1_update, loss_2_update
        
        
    def update_forget_rate(self, epoch):
        
        # TODO: UPDATE FORGET RATE FOR EVERY EPOCH
        # drop forget rate
        
        if epoch > 30:
            factor = (epoch ** self.exponent) / self.num_gradual
            self.forget_rate = self.tau * self.noise_rate * min(factor, 1)
        