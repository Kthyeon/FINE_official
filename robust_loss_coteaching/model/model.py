import torch
import torch.nn as nn
import torch.nn.functional as F
from .ResNet_Zoo import _resnet, BasicBlock, Bottleneck
from .PreActResNet_Zoo import PreActResNet, PreActBlock
from .InceptionResNetV2 import *
from torchvision.models.utils import load_state_dict_from_url

from .CIFAR_ResNet_Zoo import ResNet_s, BasicBlock_s

def inceptionresnetv2(num_classes=50):
    model = InceptionResNetV2(num_classes=num_classes)
    state_dict = load_state_dict_from_url('http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth',
                                          progress=True)
    model.load_state_dict(state_dict, strict=False)
    
    return model 

def preactresnet18(num_classes=10):
    return PreActResNet(PreActBlock, [2,2,2,2], num_classes=num_classes)


def resnet18(num_classes=10):
    return ResNet_s(BasicBlock_s, [2,2,2,2], num_classes=num_classes)

def resnet34(num_classes=10):
    return ResNet_s(BasicBlock_s, [3,4,6,3], num_classes=num_classes)

# def resnet18(pretrained=False, progress=True, num_classes=10, **kwargs):
#     r"""ResNet-18 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
# #     return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
# #                    **kwargs)
    
#     if pretrained:
#         state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet18-5c106cde.pth',
#                                               progress=progress)
#         model.load_state_dict(state_dict)
        
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs, num_classes)
        
#     return model



# def resnet34(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-34 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
#                    **kwargs)



def resnet50(pretrained=False, progress=True, num_classes=10, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    
    model = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)
    
    if pretrained:
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth',
                                              progress=progress)
        model.load_state_dict(state_dict)
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
        
    return model



def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)

def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)