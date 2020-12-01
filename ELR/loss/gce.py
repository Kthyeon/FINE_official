# https://github.com/AlanChou/Truncated-Loss/blob/master/TruncatedLoss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def singular_label(v_ortho_dict, model_represents, label):
    
    
    sing_lbl = torch.zeros(model_represents.shape[0]) == 0.
    
    for i, data in enumerate(model_represents):
        if torch.dot(v_ortho_dict[label[i].item()][0], data).abs() < torch.dot(v_ortho_dict[label[i].item()][1], data).abs():
            sing_lbl[i] = False
        
    return sing_lbl



class GCELoss(nn.Module):

    def __init__(self, q=0.7, k=0.5, trainset_size=50000, truncated=False):
        super().__init__()
        self.q = q
        self.k = k
        self.truncated = truncated
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False)
             
    def forward(self, logits, targets, indexes, tea_logits=None, model_represents = None, tea_represents=None, singular_dict=None, v_ortho_dict = None, kd = False, mode='ce'):
        if kd:
            mse = nn.MSELoss()
            ce = nn.CrossEntropyLoss(reduction='none')
            sing_lbl = singular_label(v_ortho_dict, model_represents, targets)
            
        p = F.softmax(logits, dim=1)

        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
            
        if self.truncated == True:
            loss = ((1-(Yg**self.q))/self.q)*self.weight[indexes] - ((1-(self.k**self.q))/self.q)*self.weight[indexes]
            loss = torch.mean(loss)
        else:
#             loss = (1-(Yg**self.q)) / self.q
            if mode =='ce':
                selected_p = F.softmax(logits[sing_lbl], dim=1)
                target_selected_p = torch.gather(selected_p, 1, torch.unsqueeze(targets[sing_lbl], 1)).squeeze()
                mis_p = target_selected_p < 0.01
                entropy = - torch.sum(selected_p * torch.log(selected_p), dim=1)
                ce_loss = ce(logits[sing_lbl], targets[sing_lbl])
                loss = torch.where(mis_p, torch.zeros_like(ce_loss), ce_loss)
                loss = torch.mean(loss)
            elif mode == 'same':
                selected_p = F.softmax(logits[sing_lbl], dim=1)
                target_selected_p = torch.gather(selected_p, 1, torch.unsqueeze(targets[sing_lbl], 1)).squeeze()
                mis_p = target_selected_p < 0.01
                entropy = - torch.sum(selected_p[mis_p] * torch.log(selected_p[mis_p]), dim=1)
                gce_loss = (1-(Yg**self.q)) / self.q
                loss = torch.where(mis_p, torch.zeros_like(gce_loss), gce_loss)
                loss = torch.mean(loss)
            else:
                _, prediction = torch.max(tea_logits[sing_lbl], dim=1)
                filtered = prediction == targets[sing_lbl]
                loss = ce(logits[sing_lbl][filtered], targets[sing_lbl][filtered])
                loss = torch.mean(loss)


        return loss

