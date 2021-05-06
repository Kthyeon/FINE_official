from .elr import ELRLoss, ELR_GTLoss
from .gce import GCELoss, GCE_GTLoss
from .sce import SCELoss, SCE_GTLoss
from .entropy import Entropy
from .cce import CCELoss, CCE_GTLoss
from .coteach import CoteachingLoss, CoteachingPlusLoss, CoteachingDistillLoss, CoteachingPlusDistillLoss
from torch.nn import CrossEntropyLoss