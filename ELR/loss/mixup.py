import torch
import copy
import numpy
from torch import cuda, nn
from torch.autograd import Variable
import numpy as np
__all__ = ['mixup']

def mixup(images, labels, device, tea_pred = None, alpha=1.0):
    """
    mixup function from 'mixup: BEYOND EMPIRICAL RISK MINIMIZATION', 
    https://arxiv.org/pdf/1710.09412.pdf
    """
        
    lam = numpy.random.beta(alpha, alpha)
    rand_index = torch.randperm(images.size()[0]).to(device)
    labels1 = labels
    if tea_pred != None:
        labels2 = tea_pred[rand_index]
    else:
        labels2 = labels[rand_index]
    images2 = copy.deepcopy(images)
            
    images = Variable(lam * images + (1-lam)*images2[rand_index,:,:,:]).to(device)
    
    return lam, images, labels1, labels2
    
