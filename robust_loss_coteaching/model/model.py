import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from .ResNet_Zoo import ResNet, BasicBlock, Bottleneck
from .PreActResNet_Zoo import PreActResNet, PreActBlock
from .resnet_cifar import resnet32

def preactresnet18(num_classes=10):
    return PreActResNet(PreActBlock, [2,2,2,2], num_classes=num_classes)

def resnet10(num_classes=10):
    return ResNet(BasicBlock, [1,1,1,1], num_classes=num_classes)

def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)

def resnet34(num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)

def resnet50(num_classes=10):
    import torchvision.models as models
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft

def resnet101(num_classes=10):
    return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes)

# def resnet50(num_classes=10):
#     return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes)