# https://github.com/HanxunH/SCELoss-Reproduce/blob/master/loss.py
import torch
import torch.nn.functional as F
import math
import torch.nn as nn

__all__ = ['CCELoss', 'CCE_GTLoss']

class CCELoss(nn.Module):
    def __init__(self):
        super(CCELoss, self).__init__()
        
    def forward(self, output, target, index, mode=None):
        return F.cross_entropy(output, target)
    
class CCE_GTLoss(nn.Module):
    def __init__(self):
        super(CCE_GTLoss, self).__init__()
        
    def forward(self, logits, labels, clean_indexs, mode, index=None):
        
        # index : redundant variable. This is only used in ELR.
        size = logits.shape[0] if torch.sum(clean_indexs) == 0 else torch.sum(clean_indexs)
        loss = torch.sum(F.cross_entropy(logits, labels, reduction='none')[clean_indexs]) / size
        return loss