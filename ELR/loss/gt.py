import torch
import torch.nn.functional as F
import torch.nn as nn

class GroundTruthLoss(nn.Module):
    def __init__(self, noise_indx, num_examp):
        super().__init__()
        
        self.noise_indx = noise_indx
#         self.clean_indx = [i for i in range(num_examp) if i not in noise_indx]
        
    def forward(self, logits, labels, clean_indexs):
#         clean_indexs = [i for i in range(len(indexs)) if indexs[i] not in self.noise_indx]
#         print (len(clean_indexs), sum(clean_indexs))
#         print (sum(clean_indexs))
        
        loss = torch.sum(F.cross_entropy(logits, labels, reduction='none')[clean_indexs]) / logits.shape[0]
        
        return loss