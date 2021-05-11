# https://github.com/HanxunH/SCELoss-Reproduce/blob/master/loss.py
import torch
import torch.nn.functional as F
import math
import torch.nn as nn

__all__=['SCELoss', 'SCE_GTLoss']

class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.A = math.exp(-4)
        
    def forward(self, pred, labels, index=None, mode = None):
        # index is redundant input for SCELoss
        
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=self.A, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        if mode == 'ce':
            loss = ce
        else:
            loss = self.alpha * ce + self.beta * rce.mean()
        return loss
    
class SCE_GTLoss(SCELoss):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCE_GTLoss, self).__init__(alpha, beta, num_classes)
    
    def forward(self, pred, labels, clean_indexs, index=None):
        
        # index : redundant variable. This is only used in ELR.
        
        # CE
        ce = F.cross_entropy(pred, labels, reduction='none')[clean_indexs]
        
        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=self.A, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))[clean_indexs]
        
        size = logits.shape[0] if torch.sum(clean_indexs) == 0 else torch.sum(clean_indexs)
        loss = self.alpha * torch.sum(ce) + self.beta * torch.sum(rce)
        loss /= size
        return loss
        