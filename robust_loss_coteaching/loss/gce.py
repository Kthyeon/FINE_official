# https://github.com/AlanChou/Truncated-Loss/blob/master/TruncatedLoss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

__all__=['GCELoss', 'GCE_GTLoss']

class GCELoss(nn.Module):

    def __init__(self, q=0.7, k=0.5, trainset_size=50000, truncated=False):
        super().__init__()
        self.q = q
        self.k = k
        self.truncated = truncated
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False)
             
    def forward(self, logits, targets, indexes, mode=None):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        
        if self.truncated == True:
            if mode == 'ce':
                ce = nn.CrossEntropyLoss(reduction='none')
                loss = ce(logits, targets)
                loss = torch.mean(loss)
            else:
                loss = ((1-(Yg**self.q))/self.q)*self.weight[indexes] - ((1-(self.k**self.q))/self.q)*self.weight[indexes]
                loss = torch.mean(loss)
        else:
            if mode == 'ce':
                ce = nn.CrossEntropyLoss(reduction='none')
                loss = ce(logits, targets)
                loss = torch.mean(loss)
            else:
                loss = (1-(Yg**self.q))/self.q
                loss = torch.mean(loss)

        return loss

    def update_weight(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        Lq = ((1-(Yg**self.q))/self.q)
        Lqk = np.repeat(((1-(self.k**self.q))/self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.cuda.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1)
        
        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.cuda.FloatTensor)

class GCE_GTLoss(GCELoss):
    def __init__(self, q=0.7, k=0.5, trainset_size=50000, truncated=False):
        super().__init__(q, k, trainset_size, truncated)
        
    def forward(self, logits, targets, clean_indexs, index=None):
        
        # index : redundant variable. This is only used in ELR.
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        size = logits.shape[0] if torch.sum(clean_indexs) == 0 else torch.sum(clean_indexs)
#         print (torch.mean(((1-(Yg**self.q))/self.q)))
        
        loss = (1-(Yg**self.q))/self.q
        loss = torch.sum(loss[clean_indexs]) / size
        
        return loss