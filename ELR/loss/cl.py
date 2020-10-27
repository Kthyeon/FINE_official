import torch.nn.functional as F
import torch
from parse_config import ConfigParser
import torch.nn as nn

def CrossEntropyLoss(output, target):
    return F.cross_entropy(output, target)

#일단 softhinge만 사용하는 거로
def partial_opt(margin, threshold):
    L = 0
    loss_value = SoftHingeLoss(margin, output, target) #calculate soft hing loss value
    batch_size = len(margin)
    v_set = torch.empty(batch_size)
    sorted_margin, index = torch.sort(margin) #Sort in non-dereasing order
                        
    for i in range(batch_size):
        L += sorted_margin[i]
        if L <= (threshold + 1 - i):
            v_set[index[i]] == 1
        else:
            v_set[index[i]] == 0
            
    v_set = v_set.long()
    
    return v_set

def SoftHingeLoss(margin, output, target): #margin.shape = [batch_size]
    
    soft_hinge_loss = torch.where(margin > 0, 1 - margin, 1 - output[range(len(output)), target] + torch.logsumexp(output, dim=1))
    soft_hinge_loss[soft_hinge_loss < 0] = 0 #soft_hinge_loss.shape = [batch_size]

    return soft_hinge_loss

def HardHingeLoss(margin): # margin.shape = [batch_size]
    
    hard_hinge_loss = 1 - margin
    hard_hinge_loss[hard_hinge_loss < 0] = 0

    return hard_hinge_loss


class NPCLoss(nn.Module):	
    def __init__(self, epsilon):	   
        super().__init__()	
        self.epsilon = epsilon	

    def forward(self, output, target):	
        # set base loss function	
#         base_loss = SoftHingeLoss()
        
        # margin for each data point = t_y - max(i!=y)t_i , y is target class num, shape (Batch_size,)
        tmp_output = output.clone()
        target = target.long()
        tmp_output[range(len(output)), target] = torch.max(tmp_output, dim=1).values
        
        #calculate threshold
        batch_size = output.shape[0]
        threshold = ((1 - self.epsilon) ** 2) * batch_size + (1 - self.epsilon) * torch.sum(torch.ones(batch_size)[margin < 0])
        
        threshold = int(threshold) #threshold 왜 int???

        # parameters required to calculate NPCL
        v = partial_opt(margin, threshold)
        l = SoftHingeLoss(margin)

        # calculate NPCL
        npcl_1 = torch.dot(v, l)
        npcl_2 = threshold - torch.sum(v)

        return npcl_1 if npcl_1 < npcl_2 else npcl_2
    
class CLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, output, target):
        ''' Check
        if epoch < 10:
            base_loss = HardHingeLoss()
        else:
            base_loss = SoftHingeLoss()
        '''
        tmp_output = output.clone()
        target = target.long()
        tmp_output[range(len(output)), target] = float("-inf")
        
        # margin for each data point = t_y - max(i!=y)t_i , y is target class num, shape (Batch_size,)
        margin = output[range(len(output)), target] - torch.max(tmp_output, dim=1).values
        
        # parameters required to calculate curriculum loss
        v = partial_opt(margin, threshold) # shape = [batch] / 0 or 1
        l = SoftHingeLoss(margin) # shape = [batch]
        batch_size = output.shape[0]
        
        # curriculum loss is maximum value between loss 1 and 2
        curriculum_loss_1 = torch.dot(v, l)
        curriculum_loss_2 = batch_size - torch.sum(v) + torch.sum(torch.ones(batch_size)[margin < 0])
        
        return curriculm_loss_1 if curriculm_loss_1 < curriculum_loss_2 else curriculm_loss_2
    
class Tight_CLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, output, target):
        ''' Check
        if epoch < 10:
            base_loss = HardHingeLoss()
        else:
            base_loss = SoftHingeLoss()
        '''
        tmp_output = output.clone()
        target = target.long()
        tmp_output[range(len(output)), target] = float("-inf")
        
        # margin for each data point = t_y - max(i!=y)t_i , y is target class num, shape (Batch_size,)
        margin = output[range(len(output)), target] - torch.max(tmp_output, dim=1).values
        
        # parameters required to calculate curriculum loss
        v = partial_opt(margin, threshold) # shape = [batch] / 0 or 1
        l = SoftHingeLoss(margin) # shape = [batch]
        batch_size = output.shape[0]
        
        # curriculum loss is maximum value between loss 1 and 2
        curriculum_loss_1 = torch.dot(v, l)
        curriculum_loss_2 = batch_size - torch.sum(v)
        
        return curriculm_loss_1 if curriculm_loss_1 < curriculum_loss_2 else curriculm_loss_2
    
