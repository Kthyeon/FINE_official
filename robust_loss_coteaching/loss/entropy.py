import torch
import torch.nn.functional as F
import torch.nn as nn

__all__=['Entropy']

class Entropy(nn.Module):
    def __init__(self, threshold = 0.1):
        super().__init__()
        self.threshold = threshold
        
    
    def forward(self, logits, targets, sing_lbl=None):
        if sing_lbl != None:
            num = logits.shape[0]
            p = F.softmax(logits[sing_lbl], dim=1)
            target_p = torch.gather(p, 1, torch.unsqueeze(targets[sing_lbl], 1)).squeeze()
        else:
            num = sing_lbl.sum()
            p = F.softmax(logits, dim=1)
            target_p = torch.gather(p, 1, torch.unsqueeze(targets, 1)).squeeze()
        
        entropy = - torch.sum(p * torch.log(p), dim=1)
        
        return torch.sum(entropy[target_p < thereshold]) / num
