import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from .ResNet_Zoo import ResNet, BasicBlock
from .PreActResNet import PreActResNet, PreActBlock
from .CoteachingNet import CNN_small, CNN

def cifarnet_small(num_classes=10):
    return CNN_small(num_classes=10)

def cifarnet_large(num_classes=10):
    return CNN(input_channel=3, n_outputs=num_classes, dropout_rate=0.25, momentum=0.1)

def preact_resnet18(num_classes=10):
    return PreActResNet(PreActBlock, [2,2,2,2], num_classes=num_classes)

def resnet34(num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)




