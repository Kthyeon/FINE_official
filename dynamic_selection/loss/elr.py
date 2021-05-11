import torch.nn.functional as F
import torch
from utils.parse_config import ConfigParser
import torch.nn as nn

__all__ = ['ELRLoss']

class ELRLoss(nn.Module):
    def __init__(self, num_examp, num_classes=10, beta=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.config = ConfigParser.get_instance()
        self.USE_CUDA = torch.cuda.is_available()
        self.target = torch.zeros(num_examp, self.num_classes).cuda() if self.USE_CUDA else torch.zeros(num_examp, self.num_classes)
        self.beta = beta
        
    def forward(self, output, label, index, mode = None):
        y_pred = F.softmax(output,dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
        y_pred_ = y_pred.data.detach()
        self.target[index] = self.beta * self.target[index] + (1-self.beta) * ((y_pred_)/(y_pred_).sum(dim=1,keepdim=True))
        ce_loss = F.cross_entropy(output, label)
        elr_reg = ((1-(self.target[index] * y_pred).sum(dim=1)).log()).mean()
        if mode == 'ce':
            final_loss = ce_loss
        else:
            final_loss = ce_loss +  self.config['train_loss']['args']['lambda']*elr_reg
        return  final_loss
    
class ELR_GTLoss(ELRLoss):
    def __init__(self, num_examp, num_classes, beta=0.3):
        super(ELR_GTLoss, self).__init__(num_examp, num_classes, beta)
        
    def forward(self, output, label, clean_indexs, index):
        y_pred = F.softmax(output, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
        y_pred_ = y_pred.data.detach()
        self.target[index] = self.beta * self.target[index] + (1-self.beta) * ((y_pred_)/(y_pred_).sum(dim=1,keepdim=True))
        
        ce_loss = F.cross_entropy(output, label, reduction='none')[clean_indexs]
        elr_reg = ((1-(self.target[index] * y_pred).sum(dim=1)).log())[clean_indexs]
        size = logits.shape[0] if torch.sum(clean_indexs) == 0 else torch.sum(clean_indexs)
        
        final_loss = torch.sum(ce_loss) + self.config['train_loss']['args']['lambda'] * torch.sum(elr_reg)
        final_loss /= size
        return final_loss