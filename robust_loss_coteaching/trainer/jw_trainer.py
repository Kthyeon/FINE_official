import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import List
import math
from torchvision.utils import make_grid

from .default_trainer import DefaultTrainer
from utils import inf_loop

class JongwooTrainer(DefaultTrainer):
    """
    JongwooKo's Trainer class
    
    Note:
        Inherited from BaseTrainer.
    """
    
    def __init__(self, model, train_criterion, metrics, optimizer, config, data_loader, parse,
                 valid_data_loader=None, test_data_loader=None, lr_scheduler=None, len_epoch=None, val_criterion=None):
        super().__init__(model, train_criterion, metrics, optimizer, config, data_loader, parse,
                         valid_data_loader=valid_data_loader,
                         test_data_loader=test_data_loader,
                         lr_scheduler=lr_scheduler,
                         len_epoch=len_epoch,
                         val_criterion=val_criterion)
        # val loss without reduction (label_correction)
        self.lc_criterion = nn.CrossEntropyLoss(reduction='none')
    
    def _train_epoch(self, epoch):
        """
        
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        
        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log befor return. i.e.
                > log = {**log, **additional_log}
                > return log
                
            The metrics in log must have the key 'metrics'.
        """
        # TODO: semi-supervised에서 batch를 나누는 거 같은데
        # 그거 어떻게 하는지 semi-supervised loss를 좀 살펴보도록 하자.
        
        if epoch > 1:
#         if epoch > 10 and (epoch+1) % 5 == 0: # criteria epoch
            # TODO:
            # loss를 다시 계산을 해서
            # class별로 small loss 애들을 이용하여 representation vec 만들고
            # 그걸 이용해서 pseudo label 다시 진행한 다음에
            # pseudo label하고 noisy label을 비교해서
            # 어떤 loss를 사용할지 계산을 한다.
            self.model.eval()
            
            small_loss_instances = [[] for _ in range(10)] # num_classes
            small_loss = [[] for _ in range(10)]
            clean_idxs, noise_idxs, neither_idxs = [], [], []
            
            with torch.no_grad():
                with tqdm((self.data_loader)) as progress:
                    for batch_idx, (data, label, indexs, _) in enumerate(progress):
                        data, label = data.to(self.device), label.long().to(self.device)

                        output = self.model(data)
                        prob = torch.softmax(output, dim=-1)
                        loss = self.lc_criterion(output, label).detach() # 이거 reduce하면 안됨. 다시 살펴볼것.

                        # class 별로 small loss instance 정리
                        # small loss criterion
                        for x, y, z in zip(loss, prob, label):
                            small_loss[z].append(x.cpu().item())
                            small_loss_instances[z].append(y.cpu().numpy().tolist())

                # respresentation vec
                # TODO: List length Balancing
                maxlen = max([len(x) for x in small_loss])
                for i in range(len(small_loss)):
                    small_loss[i] += [math.inf for _ in range(maxlen-len(small_loss[i]))]
                    small_loss_instances[i] += [[0.1 for _ in range(10)] for _ in range(maxlen-len(small_loss_instances[i]))]
    
                # Index slicing
                small_loss = -torch.Tensor(small_loss)
                small_loss_instances = torch.Tensor(small_loss_instances)
                
                _, small_loss_idx = small_loss.topk(50, dim=-1)
                
#                 small_loss_instances = torch.gather(small_loss_instances, 2, small_loss_idx)
#                 small_loss_instances = torch.index_select(small_loss_instances, 1, small_loss_idx)
                small_loss_instances = torch.Tensor([small_loss_instances[i, small_loss_idx[i]].numpy().tolist() for i in range(10)])
    
                # Mean of small loss instances
                r = torch.softmax(small_loss_instances, dim=-1).mean(dim=1)
                r = r.unsqueeze(1).to(self.device).detach()
                
                # pseudo labeling
                
                with tqdm((self.data_loader)) as progress:
                    for batch_idx, (data, label, indexs, _) in enumerate(progress):
                        data, label = data.to(self.device), label.long().to(self.device)
                        
                        # instance와 거리 비교
                        output = self.model(data).detach()
                        output_prob = torch.softmax(output, dim=-1).detach()
                        
                        # TODO: noise label과 pseudo label 비교. confidence도 비교.
                        # pseudo label 구함.
                        output_prob_ = output_prob.unsqueeze(0)
                        neg_squared_diff = -((r-output_prob_).pow(2).sum(-1))
                        pseudo_label = neg_squared_diff.topk(1, dim=0)[1].squeeze(0)
                        
                        # noise label과 pseudo label 비교.
                        # confidence for pseudo label  
#                         pseudo_conf = torch.gather(output_prob, 1, pseudo_label)
                        pseudo_conf = torch.Tensor([output_prob[i, pseudo_label[i]].item() for i in range(len(label))])
                        
                        for x, y, z, w in zip(label, pseudo_label, pseudo_conf, indexs):
                            if x == y and z > 0.5: # 0.5는 hyperparameter. 이론상은 0.5지만 더 높아질 수 있을것으로 보임.
                                clean_idxs.append(w.item())
                            elif x != y and z < 0.1: # 이거도 현재 이론상은 0.1인데, 더 높아질 수 있을것으로 보임.
                                noise_idxs.append(w.item())
                            else:
                                neither_idxs.append(w.item())
                                
        # TODO: DATA LOADER 인덱스 별로 쪼개서 나눠서 학습하게 하는 거 만들것.
        # TODO: 이 씨발련들 왜 학습이 안돼.....
        # TODO: 아무것도 안건드렸다고...
                
            self.clean_dataset = torch.utils.data.Subset(self.data_loader.dataset, clean_idxs)
            self.noise_dataset = torch.utils.data.Subset(self.data_loader.dataset, noise_idxs)
            self.dataset = torch.utils.data.Subset(self.data_loader.dataset, neither_idxs)
                                
        return super()._train_epoch(epoch)
    
    def _warmup_epoch(self, epoch):
        # ELR plus에서만 사용하는 함수.
        pass