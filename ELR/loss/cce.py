# https://github.com/HanxunH/SCELoss-Reproduce/blob/master/loss.py
import torch
import torch.nn.functional as F
import math
import torch.nn as nn

class CCELoss(nn.Module):
    def __init__(self):
        super(CCELoss, self).__init__()
        
    def forward(self, output, target, index):
        return F.cross_entropy(output, target)
    
class GTLoss(nn.Module):
    def __init__(self):
        suepr(GTLoss, self).__init__()
        
    def forward(self, logits, labels, clean_indexs):
        loss = torch.sum(F.cross_entropy(logits, labels, reduction='none')[clean_indexs]) / logits.shape[0]
        return loss