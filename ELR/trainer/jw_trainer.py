import torch
import numpy as np
from tqdm import tqdm
from typing import List
from torchvision.utils import make_grid

from base import BaseTrainer
from utils import inf_loop

class JongwooTrainer(BaseTrainer):
    """
    JongwooKo's Trainer class
    
    Note:
        Inherited from BaseTrainer.
    """
    
    def __init__(self, model, train_criterion, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, test_data_loader=None, lr_scheduler=None, len_epoch=None, val_criterion=None):
        super().__init__(model, train_criterion, metrics, optimizer, config, val_criterion)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epcoh-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteraion-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        
        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.do_test = self.test_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.train_loss_list: List[float] = []
        self.val_loss_list: List[float] = []
        self.test_loss_list: List[float] = []
            
    def _eval_metrics(self, output, label):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, label)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics
    
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
        
        self.model.train()
        
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        total_metrics_gt = np.zeros(len(self.metrics))
        
        with tqdm(self.data_loader) as progress:
            for batch_idx, (data, label, indexs, gt) in enumerate(progress):
                progress.set_description_str(f'Train epoch {epoch}')
                
                data, label = data.to(self.device), label.long().to(self.device)
                gt = gt.long().to(self.device)
                
                output = self.model(data)
                loss = self.train_criterion(output, label, indexs.cpu().detach().numpy().tolist()) # TODO: 어떤 instances인지에 따라 서로 다른 loss를 사용해야 함.
                self.optimizer.zero_grad()
                loss.backward()
                
                self.optimizer.step()
                
#                 self.writer.set_step((epoch - 1) *)
        
        
        if epoch > 10 and (epoch+1) % 5 == 0: # criteria epoch
            # TODO:
            # loss를 다시 계산을 해서
            # class별로 small loss 애들을 이용하여 representation vec 만들고
            # 그걸 이용해서 pseudo label 다시 진행한 다음에
            # pseudo label하고 noisy label을 비교해서
            # 어떤 loss를 사용할지 계산을 한다.
            
            small_loss_logits = [[] for _ in range(10)] # num_classes
            
            with torch.no_grad():
                with tqdm((self.data_loader)) as progress:
                    for batch_idx, (data, label, indexs, _) in enumerate(progress):
                        data, label = data.to(self.device), label.long().to(self.device)

                        output = self.model(data)
                        loss = self.val_criterion(output, label)

                        # class 별로 small loss instance 정리
                        # small loss criterion
                        

                # respresentation vec
                r = torch.softmax(torch.Tensor(small_loss_logits), dim=-1).mean(dim=0)
                
                # pseudo labeling
                with tqdm((self.data_loader)) as progress:
                    for batch_idx, (data, label, indexs, _) in enumerate(progress):
                        data, label = data.to(self.device), label.long().to(self.device)
                        
                        # instance와 거리 비교
                        output = self.model(data)
                        output = torch.softmax(output, dim=-1)
                        
                        # noise label과 pseudo label 비교.
                        # confidence도 비교.
                        