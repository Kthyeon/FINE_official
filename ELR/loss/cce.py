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