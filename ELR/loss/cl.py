import torch.nn.functional as F
import torch
from parse_config import ConfigParser
import torch.nn as nn

def CrossEntropyLoss(output, target):
    return F.cross_entropy(output, target)


class ELRLoss(nn.Module):
    def __init__(self, num_examp, num_classes=10, beta=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.config = ConfigParser.get_instance()
        self.USE_CUDA = torch.cuda.is_available()
        self.target = torch.zeros(num_examp, self.num_classes).cuda() if self.USE_CUDA else torch.zeros(num_examp, self.num_classes)
        self.beta = beta
        
    def forward(self, output, label, index):
        y_pred = F.softmax(output,dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
        y_pred_ = y_pred.data.detach()
        self.target[index] = self.beta * self.target[index] + (1-self.beta) * ((y_pred_)/(y_pred_).sum(dim=1,keepdim=True))
        ce_loss = F.cross_entropy(output, label)
        elr_reg = ((1-(self.target[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss +  self.config['train_loss']['args']['lambda']*elr_reg
        return  final_loss
    
class NPCLoss(nn.Module):

class SoftHingeLoss(nn.Module):
    def __init__(self):
        super(SoftHingeLoss, self).__init__()

    def forward(self, output, target):
        tmp_output = output.clone()
        target = target.long()
        tmp_output[range(len(output)), target] = float("-inf")
        
        # margin for each data point = t_y - max(i!=y)t_i , y is target class num
        margin = output[range(len(output)), target] - torch.max(tmp_output, dim=1).values
        
        # soft hinge loss described in the paper elr, appendix H
        if margin >= 0:
            soft_hinge_loss = 1 - margin
        else:
            soft_hinge_loss = 1 - output[range(len(output)), target] + torch.logsumexp(output, dim=1)
        
        soft_hinge_loss[soft_hinge_loss < 0] = 0
        
        return soft_hinge_loss
    